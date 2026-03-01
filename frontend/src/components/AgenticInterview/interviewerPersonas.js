/**
 * interviewerPersonas.js
 * 4 rotating interviewer personas.
 * Avatar rendering is handled by AvatarViewer.js using the Three.js public CDN.
 * Each persona applies a different accent colour tint + distinct namecard info.
 */

export const PERSONAS = [
  {
    id: 0,
    name: 'Emily Chen',
    title: 'Senior Software Engineer',
    company: 'Google',
    companyColor: '#4285F4',
    initials: 'EC',
    avatarColor: '#4338CA',   // indigo
    accentColor: '#818CF8',
    style: 'thorough and detail-oriented',
  },
  {
    id: 1,
    name: 'Marcus Reid',
    title: 'Tech Lead',
    company: 'Amazon',
    companyColor: '#FF9900',
    initials: 'MR',
    avatarColor: '#92400E',   // amber-dark
    accentColor: '#FCD34D',
    style: 'direct and fast-paced',
  },
  {
    id: 2,
    name: 'Priya Sharma',
    title: 'Engineering Manager',
    company: 'Microsoft',
    companyColor: '#00A4EF',
    initials: 'PS',
    avatarColor: '#0369A1',   // sky-dark
    accentColor: '#38BDF8',
    style: 'system-design focused',
  },
  {
    id: 3,
    name: 'David Okafor',
    title: 'Staff Engineer',
    company: 'Meta',
    companyColor: '#1877F2',
    initials: 'DO',
    avatarColor: '#5B21B6',   // violet
    accentColor: '#A78BFA',
    style: 'algorithm and complexity focused',
  },
];

/**
 * Returns the persona for the current session, cycling through all 4.
 * @param {number} sessionCount  — increments each interview (stored in localStorage)
 */
export const getPersona = (sessionCount = 0) => PERSONAS[sessionCount % PERSONAS.length];
