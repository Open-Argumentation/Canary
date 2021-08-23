import json
import os

import canary

cache_folder = ".cache"


class Cache:

    def __init__(self):
        self.cache_location = canary.utils.CANARY_LOCAL_STORAGE / cache_folder
        os.makedirs(self.cache_location, exist_ok=True)

    def cache(self, name, data):
        name = self.cache_location / name
        with open(name, 'w') as f:
            json.dump(data, f)

        return name

    def load(self, name):
        name = self.cache_location / name
        try:
            with open(name, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            canary.utils.logger.warn(f"Could not find {name.name} in the cache")
