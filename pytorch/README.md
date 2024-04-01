## Folder Structure

* `config/*_config.json`: JSON configuration file
* `neural_networks/*_NN.py`: Neural network architecture
* `*_train.py`: Training script
* `*_eval.py`: Evaluation script
* `models/*_model_weights.pth`: model state dict exported
  with `torch.save(model.state_dict(), "models/*_model_weights.pth")` (cleaner than loading complete model)
* `models/*_model.pth`: complete model exported with `torch.save(model, "models/*_model.pth")`

## Running scripts
### Training 
`python *_train.py "config/*_config.json"`
### Evaluation 
`python *_eval.py "config/*_config.json"`

## TODO

- [x] Setup JSON file argument parsing
- [ ] Unit tests