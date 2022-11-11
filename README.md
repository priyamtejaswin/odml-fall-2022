# SpeechCommands PyTorch CNN Baseline

* Pre-trained state dictionary -- `sc_v01_trained.pt`
* All data -- <https://huggingface.co/datasets/speech_commands/blob/main/dataset_infos.json>
* Test data -- `test/`
* Model class -- spec_cnn_baseline.py
* Evaluate
    * Env -- `source ~/whisper_env/bin/activate`
    * Command -- `python evalbaseline.py path/to/dir/with/files/in/label/dirs.wav`