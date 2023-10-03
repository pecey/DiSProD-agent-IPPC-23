# DiSProD for pyRDDLGym


## Getting started

### Prerequisites
Install the [pyRDDLGym](https://github.com/ataitler/pyRDDLGym) project. This can be done via pip.
```bash
pip install pyRDDLGym==1.0.4
```
We use the RDDL parsing logic from pyRDDLGym v1.0.4. So, newer versions of pyRDDLGym may or may not work depending on if the existing parsing logic works. 

The code for the parsing logic is present in `pyRDDLGymHelper/Core` folder.

### Dependencies
```bash
pip install -r requirements.txt
```

## Running against a PyRDDL environment.

```bash
python run_gym.py <domain_name> <instance_name> <method_name> <#iterations>
```

- <domain_name> is the domain to test on.
- <instance_name> is the instance number to use.
- <method_name> is your method name, as will be logged in the results logs.
- <#iterations> is the number of episodes to test.




