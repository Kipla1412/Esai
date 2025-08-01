""" Configuration Module """

import json 
import os
from ...serialize import SerializeFactory

class Configuration :

    def load(self,path):

        config = None

        jsonconfig = os.path.exists(f"{path}/config.json")

        name = "config.json" if jsonconfig else "config"

        with open(f"{path}/{name}" ,"r" if jsonconfig else "rb" , encoding="utf-8" if jsonconfig else None) as handle:

            config = json.load(handle) if jsonconfig else SerializeFactory.create("pickle") .loadstream(handle)

            config["format"] = "json" if jsonconfig else "pickle"

        return config

    def save(self,config,path):

        jsonconfig = config.get("format","json") == "json"

        name = "config.json" if jsonconfig else "config"

        with open(f"{path}/{name}",'w' if jsonconfig else 'wb',encoding="utf-8" if jsonconfig else None) as handle :

            if jsonconfig:

                json.dump(config,handle,default=str,intent=2)

            else :

                SerializeFactory.create("pickle").savestream(config,handle)