class AudioIOWorklet extends AudioWorkletProcessor {
    constructor(options) {
        super();
        this.port.onmessage = this.onmessage.bind(this);

    }

    process(inputList, outputList, parameters) {
        const input = inputList[0];
        const output = outputList[0];
        // TODO: Finish this
        this.port.postMessage({"action":"inputQuanta", "input":input});
        
        return true;
    }

    onmessage(evt) {
        console.log("received", evt.data.test, "in audio worklet");
    }
}
registerProcessor("audio-io-worklet", AudioIOWorklet);