class PCMProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = [];
        this.bufferSize = 1600; // 100 ms @ 16kHz
    }

    process(inputs) {
        const input = inputs[0];
        if (!input || !input[0]) return true;

        const channelData = input[0];

        for (let i = 0; i < channelData.length; i++) {
            this.buffer.push(channelData[i]);
        }

        // 🔥 Send only when enough audio is accumulated
        if (this.buffer.length >= this.bufferSize) {
            const chunk = this.buffer.slice(0, this.bufferSize);
            this.buffer = this.buffer.slice(this.bufferSize);

            // Amplify + clamp
            const amplified = new Float32Array(chunk.length);
            for (let i = 0; i < chunk.length; i++) {
                amplified[i] = Math.max(-1, Math.min(1, chunk[i] * 4));
            }

            this.port.postMessage(amplified);
        }

        return true;
    }
}

registerProcessor('pcm-processor', PCMProcessor);
