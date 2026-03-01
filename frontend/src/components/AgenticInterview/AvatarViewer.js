/**
 * AvatarViewer.js
 * 3D interviewer avatar using React Three Fiber.
 * Falls back gracefully to an animated CSS avatar when the 3D model fails to load.
 *
 * Avatar source: Three.js public CDN (CORS enabled, persistent) + Mixamo-converted GLBs
 * hosted on GitHub's CDN (raw.githubusercontent.com).
 *
 * Props:
 *  - persona    {object}  Interviewer persona from interviewerPersonas.js
 *  - isSpeaking {bool}    True when the AI interviewer is speaking
 */

import React, { Suspense, useRef, useEffect, useState, useCallback } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { useGLTF, OrbitControls, Environment, ContactShadows } from '@react-three/drei';
import * as THREE from 'three';

// ─── Procedural animation settings ──────────────────────────────────────────
const IDLE_SPEED = 0.55;
const TALK_SPEED = 2.4;

// ─── Reliable public GLB models (Three.js official CDN) ──────────────────────
// These are the example character models from the Three.js team — always online.
export const PUBLIC_AVATAR_URLS = [
  // Michelle — female, realistic, professional-looking animated model
  'https://threejs.org/examples/models/gltf/Michelle.glb',
  // RobotExpressive — friendly robot as a fallback to a second 3D model
  'https://threejs.org/examples/models/gltf/Michelle.glb',
  // Repeat Michelle for the other personas (different lighting/color per persona)
  'https://threejs.org/examples/models/gltf/Michelle.glb',
  'https://threejs.org/examples/models/gltf/Michelle.glb',
];

// ─── 3D Model inner component ─────────────────────────────────────────────────
function AvatarModel({ glbUrl, isSpeaking, accentColor }) {
  const { scene, animations } = useGLTF(glbUrl);
  const mixer   = useRef(null);
  const elapsed = useRef(0);
  const headBone  = useRef(null);
  const spineBone = useRef(null);

  useEffect(() => {
    if (!scene) return;

    // Normalise scale & recentre
    const box  = new THREE.Box3().setFromObject(scene);
    const size = box.getSize(new THREE.Vector3());
    const sc   = 1.95 / size.y;
    scene.scale.setScalar(sc);
    scene.position.set(
      -box.getCenter(new THREE.Vector3()).x * sc,
      -box.min.y * sc,
      -box.getCenter(new THREE.Vector3()).z * sc
    );

    // Tint the model subtly toward persona accent colour
    const col = new THREE.Color(accentColor);
    scene.traverse(node => {
      if (node.isMesh && node.material) {
        const mats = Array.isArray(node.material) ? node.material : [node.material];
        mats.forEach(m => {
          if (m.color) m.color.lerp(col, 0.04);
        });
      }
      if (node.isBone) {
        const n = node.name.toLowerCase();
        if (n.includes('head') && !headBone.current)  headBone.current  = node;
        if (n.includes('spine') && !spineBone.current) spineBone.current = node;
      }
    });

    // Play embedded animations if present (Michelle has Walk, Run, etc.)
    if (animations?.length) {
      mixer.current = new THREE.AnimationMixer(scene);
      // pick the most "idle"-like clip
      const clip =
        animations.find(a => /idle|stand|breath/i.test(a.name)) ||
        animations.find(a => /walk/i.test(a.name)) ||
        animations[0];
      const action = mixer.current.clipAction(clip);
      action.setEffectiveWeight(0.25); // subtle — we overlay procedural anim
      action.play();
    }
  }, [scene, animations, accentColor]);

  useFrame((_, delta) => {
    elapsed.current += delta;
    const t = elapsed.current;

    if (mixer.current) mixer.current.update(delta * 0.5); // slow embedded anim

    // Spine breathe
    if (spineBone.current) {
      spineBone.current.rotation.z = Math.sin(t * IDLE_SPEED) * 0.010;
      spineBone.current.rotation.x = Math.sin(t * IDLE_SPEED * 0.6) * 0.007;
    }

    // Head motion
    if (headBone.current) {
      if (isSpeaking) {
        headBone.current.rotation.x = Math.sin(t * TALK_SPEED) * 0.07 - 0.04;
        headBone.current.rotation.y = Math.sin(t * TALK_SPEED * 0.75) * 0.055;
        headBone.current.rotation.z = Math.sin(t * TALK_SPEED * 0.55) * 0.025;
      } else {
        headBone.current.rotation.x = Math.sin(t * 0.38) * 0.018 - 0.018;
        headBone.current.rotation.y = Math.sin(t * 0.27) * 0.025;
        headBone.current.rotation.z = Math.sin(t * 0.47) * 0.010;
      }
    }

    // Jaw morph for speaking
    scene.traverse(node => {
      if (!node.morphTargetInfluences || !node.morphTargetDictionary) return;
      const idx =
        node.morphTargetDictionary['jawOpen'] ??
        node.morphTargetDictionary['mouthOpen'] ??
        node.morphTargetDictionary['viseme_aa'];
      if (idx !== undefined) {
        const target = isSpeaking ? Math.abs(Math.sin(t * 8)) * 0.45 : 0;
        node.morphTargetInfluences[idx] = THREE.MathUtils.lerp(
          node.morphTargetInfluences[idx], target, 0.18
        );
      }
    });
  });

  return <primitive object={scene} />;
}

// ─── CSS Fallback Avatar ──────────────────────────────────────────────────────
export function FallbackAvatar({ persona, isSpeaking }) {
  return (
    <div className="avatar-fallback" style={{ '--accent': persona.accentColor }}>
      <div className={`avatar-fallback-ring ${isSpeaking ? 'speaking' : ''}`}>
        <div
          className="avatar-fallback-face"
          style={{ background: `linear-gradient(135deg, ${persona.avatarColor}, ${persona.accentColor})` }}
        >
          <span className="avatar-fallback-initials">{persona.initials}</span>
          {isSpeaking && (
            <div className="avatar-sound-wave">
              {[...Array(6)].map((_, i) => (
                <span key={i} className="wave-bar" style={{ animationDelay: `${i * 0.09}s` }} />
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Orbiting particles */}
      <div className="avatar-particles">
        {[...Array(6)].map((_, i) => (
          <div
            key={i}
            className="particle"
            style={{
              '--delay': `${i * 0.65}s`,
              '--x': `${Math.cos((i * 60 * Math.PI) / 180) * 72}px`,
              '--y': `${Math.sin((i * 60 * Math.PI) / 180) * 72}px`,
            }}
          />
        ))}
      </div>
    </div>
  );
}

// ─── Three.js Canvas Loader (inner — never throws outside Suspense) ───────────
function SafeModelLoader({ glbUrl, isSpeaking, accentColor, onError }) {
  useEffect(() => {
    // Preload the URL; if it 404s the ErrorBoundary will catch it
    useGLTF.preload(glbUrl);
  }, [glbUrl]);

  return <AvatarModel glbUrl={glbUrl} isSpeaking={isSpeaking} accentColor={accentColor} />;
}

// ─── React Error Boundary ─────────────────────────────────────────────────────
class AvatarBoundary extends React.Component {
  constructor(props) { super(props); this.state = { crashed: false }; }
  static getDerivedStateFromError() { return { crashed: true }; }
  componentDidCatch(err) { console.warn('[AvatarViewer] 3D load failed, using CSS fallback:', err.message); }
  render() {
    return this.state.crashed ? this.props.fallback : this.props.children;
  }
}

// Set to true to attempt loading the Three.js GLB model.
// Currently false because the demo Michelle.glb is not professional-looking enough.
// Swap in a proper business-attire GLB and set this to true to enable 3D.
const USE_3D_AVATAR = false;

// ─── Main Component ───────────────────────────────────────────────────────────
const AvatarViewer = ({ persona, isSpeaking }) => {
  const [webGLOk, setWebGLOk] = useState(true);

  // WebGL pre-flight check (still run so it's ready when 3D is re-enabled)
  useEffect(() => {
    try {
      const c = document.createElement('canvas');
      if (!c.getContext('webgl') && !c.getContext('experimental-webgl')) {
        setWebGLOk(false);
      }
    } catch {
      setWebGLOk(false);
    }
  }, []);

  const glbUrl = PUBLIC_AVATAR_URLS[persona.id % PUBLIC_AVATAR_URLS.length];

  // Use CSS avatar as primary — swaps to 3D when USE_3D_AVATAR = true
  if (!USE_3D_AVATAR || !webGLOk) {
    return (
      <div className="avatar-canvas-wrapper">
        <FallbackAvatar persona={persona} isSpeaking={isSpeaking} />
        {isSpeaking && (
          <div className="avatar-speaking-pulse" style={{ '--accent': persona.accentColor }} />
        )}
      </div>
    );
  }


  return (
    <div className="avatar-canvas-wrapper">
      <AvatarBoundary fallback={<FallbackAvatar persona={persona} isSpeaking={isSpeaking} />}>
        <Canvas
          camera={{ position: [0, 1.5, 1.1], fov: 38 }}
          gl={{ antialias: true, powerPreference: 'high-performance' }}
          style={{ background: 'transparent' }}
          onError={() => setWebGLOk(false)}
        >
          {/* Lighting */}
          <ambientLight intensity={0.55} />
          <directionalLight position={[2, 4, 3]} intensity={1.1} castShadow />
          <directionalLight position={[-2, 2, -1]} intensity={0.35} color="#b4c8ff" />
          <pointLight position={[0, 1, 1.5]} intensity={0.4} color={persona.accentColor} />

          <Environment preset="city" />

          <Suspense fallback={null}>
            <SafeModelLoader
              glbUrl={glbUrl}
              isSpeaking={isSpeaking}
              accentColor={persona.accentColor}
            />
            <ContactShadows position={[0, -0.98, 0]} opacity={0.35} scale={3} blur={2.5} />
          </Suspense>

          <OrbitControls
            enableZoom={false}
            enablePan={false}
            minPolarAngle={Math.PI / 2 - 0.3}
            maxPolarAngle={Math.PI / 2 + 0.15}
            minAzimuthAngle={-0.3}
            maxAzimuthAngle={0.3}
            target={[0, 1.5, 0]}
          />
        </Canvas>
      </AvatarBoundary>

      {/* Speaking pulse ring */}
      {isSpeaking && (
        <div
          className="avatar-speaking-pulse"
          style={{ '--accent': persona.accentColor }}
        />
      )}
    </div>
  );
};

export default AvatarViewer;
