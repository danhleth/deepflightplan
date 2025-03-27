# Deepflightplan
A framework support generate flight plan dataset

## How to Run the pipeline

### Install Enviroments
You'll need install enviroments by file `requirements.yaml`
```
conda env create -f environments.yaml
```
### Run models
- Change directory to `tasks/generating_flightplan`
- Create an example configs like configs/general_config.yaml then run:
```
python run.py -c [path_to_config_file]
```

If you want to customize the configuration, please access this [doc](docs/config.md)