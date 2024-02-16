class ParticleFilter {
    
}


class ParticleWorkletProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        this.hop = options.processorOptions.hop;


    }

    process(inputList, outputList, parameters) {
    const sourceLimit = Math.min(inputList.length, outputList.length);
    
    for (let inputNum = 0; inputNum < sourceLimit; inputNum++) {
        const input = inputList[inputNum];
        const output = outputList[inputNum];
        const channelCount = Math.min(input.length, output.length);
    
        for (let channelNum = 0; channelNum < channelCount; channelNum++) {
            console.log(input[channelNum].length);
        input[channelNum].forEach((sample, i) => {
            // Manipulate the sample
            output[channelNum][i] = 0;//sample;
        });
        }
    };
    
    return true;
    }      
}
registerProcessor("particle-worklet-processor", ParticleWorkletProcessor);