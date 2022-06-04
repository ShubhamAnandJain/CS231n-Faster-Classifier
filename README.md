# Bag of Tricks for Faster & Stable Image Classification

This is a Python library for using the latest SOTA techniques to improve the training pipeline of DNNs, making them faster and robust.

## Installation

Use the package manager [conda](https://docs.conda.io/en/latest/) to install `torch==1.7 torchvision==0.8`

Run the following script to install custom PyTorch layers for Approximate Tensor Operations
```bash
./setup.sh
```

## Usage

Sample Notebook showing how to use the library is attached in `./src/demo.ipynb`

### Adding a New Model
Edit `./src/models.py`, and add your model to the `model_dict` in `get_model`

```python
def get_model(model_name, model_params, learning_rate, loader_train, num_channels, device):
    model_dict = {
        'VGG16' : models.vgg16, 
        'Resnet18' : models.resnet18, 
        'Resnet50' : models.resnet50, 
    }

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

If you use this repository or extend it, please cite us.


## Authors
[Aman Bansal](https://github.com/daydroidmuchiri)

[Shubham Anand Jain](https://github.com/ShubhamAnandJain)

[Bharat Khandelwal](https://github.com/khandelwalbharat)

