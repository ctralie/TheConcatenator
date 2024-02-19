class AudioIOWorklet extends AudioWorkletProcessor {
    constructor(options) {
        super();
        this.port.onmessage = this.onmessage.bind(this);
        this.outputQueue = [];
    }

    process(inputList, outputList, parameters) {
        const input = inputList[0];
        const output = outputList[0];
        for (let i = 0; i < input.length; i++) {
            output[i] = input[i];
        }
        // Send new input quanta off to be incorporated
        this.port.postMessage({"action":"inputQuanta", "input":input}); 
        // Request new output quanta
        this.port.postMessage({"action":"pullQuanta"});
        // Output the least recently pushed audio quantum if any are available
        if (this.outputQueue.length > 0) {
            let next = this.outputQueue.shift();
            for (let i = 0; i < output.length; i++) {
                output[i].set(next[i]);
            }
        }
        return true;
    }

    onmessage(evt) {
        if (evt.data.action == "pushQuanta") {
            this.outputQueue.push(evt.data.output);
        }
    }
}
registerProcessor("audio-io-worklet", AudioIOWorklet);