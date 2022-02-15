import yaml

with open('mobilenet.yml', 'r') as cf:
  cfg = yaml.safe_load(cf)
print(cfg)
