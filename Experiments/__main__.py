"""Display available experiment configurations without constructing or training models."""
import json
from Experiments.factory import catalog

def main():
    print(json.dumps(catalog(),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
