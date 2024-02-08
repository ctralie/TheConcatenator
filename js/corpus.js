class AudioCorpus {
    constructor(sr, nChannels, domStr, progressBar) {
        this.sr = sr;
        this.nChannels = nChannels;
        const domElement = document.getElementById(domStr);
        this.domElement = domElement;
        this.progressBar = progressBar;
        this.audioContext = new AudioContext({sampleRate:this.sr});
        this.audioSamples = {};

        this.setupDOM();
    }

    setupDOM() {
        const that = this;
        const domElement = this.domElement;
        let title = document.createElement("h2");
        title.innerHTML = "Corpus Manager";
        domElement.appendChild(title);

        // Step 1: Add list to manage chosen files
        this.filesList = document.createElement("select");
        domElement.appendChild(this.filesList);
        domElement.appendChild(document.createElement("p"));
        this.filesList.addEventListener("change", (e) => {
            const fileStr = that.filesList.value;
            that.progressBar.changeToReady("Examining <b>" + fileStr + "</b>");
            that.audioSamples[fileStr].connectToPlayer(that.audioControls);
        });

        // Step 2: Add audio player to examine corpus elements
        this.audioControls = document.createElement("audio");
        this.audioControls.id = "corpusControls";
        this.audioControls.controls = "controls";
        this.audioControls.style = "width: 80%;";
        this.innerHTML = "Your browser does not support the audio element.";
        domElement.appendChild(this.audioControls);
        domElement.appendChild(document.createElement("p"));

        // Step 3: Create input handler for individual files
        let label = document.createElement("span");
        label.innerHTML = "<b>Add An Audio File:</b>";
        domElement.appendChild(label);
        let tuneInput = document.createElement("input");
        domElement.appendChild(tuneInput);
        tuneInput.type = "file";
        tuneInput.addEventListener('change', function(e) {
            const fileStr = tuneInput.files[0].name;
            that.progressBar.startLoading("Loading " + fileStr);
            let reader = new FileReader();
            let audio = new SampledAudio(that.audioContext, that.sr, that.nChannels);
            reader.onload = function(e) {
                audio.setSamplesAudioBuffer(e.target.result).then(
                    ()=> {
                        that.progressBar.changeToReady("Loaded <b>" + fileStr + "</b> successfully!");
                        audio.connectToPlayer(that.audioControls);
                        that.audioSamples[fileStr] = audio;
                        const option = document.createElement("option");
                        option.value = fileStr;
                        option.innerHTML = fileStr;
                        that.filesList.appendChild(option);
                        that.filesList.value = fileStr;
                    }
                );
            }
            reader.readAsArrayBuffer(tuneInput.files[0]);
        });

        // Step 4: Create input handler for a folder
        //TODO: Fill this in
        
    }
}