class AudioIOWorklet extends AudioWorkletProcessor {
    constructor(options) {
        super();
        this.port.onmessage = this.onmessage.bind(this);
        this.outputQueue = [];
    }

    process(inputList, outputList, parameters) {
        const input = inputList[0];
        const output = outputList[0];
        // Send new input quanta off to be incorporated
        this.port.postMessage({"action":"inputQuanta", "input":input}); 
        // Request new output quanta
        this.port.postMessage({"action":"pullQuanta"});
        // Output the least recently pushed audio quantum if any are available
        if (this.outputQueue.length > 0) {
            let next = this.outputQueue.pop();
            for (let i = 0; i < output.length; i++) {
                for (let k = 0; k < next[i].length; k++) {
                    output[i][k] = next[i][k]/2;
                }
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