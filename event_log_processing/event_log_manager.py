import os

import pm4py
import pandas as pd
import pickle

from enum import Enum
from typing import Dict, Any


class EventLogParameters(Enum):

    ALLOWED_OBJECT_TYPES = "ALLOWED_OBJECT_TYPES"
    ALLOWED_ACTIVITIES = "ALLOWED_ACTIVITIES"


class EventLogManager:

    @classmethod
    def load(cls, name):
        path = os.getcwd()
        path = os.path.join(path, "event_log_processing")
        path = os.path.join(path, name + ".pkl")
        with open(path, "rb") as rf:
            return pickle.load(rf)

    def __init__(self, name):
        self.name = name
        self.__activities = None
        self.__object_types = None
        self.ocel2 = None

    def save(self):
        path = os.getcwd()
        path = os.path.join(path, "event_log_processing")
        path = os.path.join(path, self.name + ".pkl")
        with open(path, "wb") as wf:
            pickle.dump(self, wf)

    def load_ocel2_sqlite(self, path, parameters: Dict[EventLogParameters, Any] = None):
        if parameters is None:
            parameters = {}
        self.ocel2 = pm4py.read_ocel2_sqlite(path)
        if EventLogParameters.ALLOWED_OBJECT_TYPES in parameters:
            self.__object_types = parameters[EventLogParameters.ALLOWED_OBJECT_TYPES]
            self.ocel2 = pm4py.filter_ocel_object_types(self.ocel2, self.__object_types)
        else:
            self.__object_types = set(self.ocel2.objects["ocel:type"].values)
        if EventLogParameters.ALLOWED_ACTIVITIES in parameters:
            self.__activities = parameters[EventLogParameters.ALLOWED_ACTIVITIES]
            object_type_allowed_activities = {
                object_type : self.__activities
                for object_type in self.__object_types
            }
            self.ocel2 = pm4py.filter_ocel_object_types_allowed_activities(self.ocel2, object_type_allowed_activities)
        else:
            self.__activities = set(self.ocel2.events["ocel:activity"].values)

    def get_activities(self):
        return self.__activities

    def get_object_types(self):
        return self.__object_types

    def create_dataframes(self):
        self.__create_basic_frames()
        self.__create_object_evolution_frame()
        self.__create_attribute_enriched_event_frame()

    def __create_basic_frames(self):
        ocel2 = self.ocel2
        self.__event_frame = ocel2.events[:]
        self.__e2o_frame = ocel2.relations[["ocel:eid", "ocel:oid", "ocel:qualifier", "ocel:type"]][:]
        self.__o2o_frame = ocel2.o2o[:]
        self.__object_frame = ocel2.objects[:]
        self.__object_change_frame = ocel2.object_changes[:]

    def __create_object_evolution_frame(self):
        object_frame = self.__object_frame
        event_frame = self.__event_frame
        object_change_frame = self.__object_change_frame
        mintime = min(event_frame["ocel:timestamp"].values)
        changetimes = object_change_frame["ocel:timestamp"].values
        if len(changetimes) > 0:
            mintime = min(mintime, changetimes)
        max_date_string = "31.12.2099 23:59:59"
        maxtime = pd.Timestamp(max_date_string)
        object_frame["ocel:field"] = pd.NA
        object_frame["ocel:timestamp"] = mintime
        object_evolutions = pd.concat([object_frame, object_change_frame])
        object_evolutions.reset_index(drop=True, inplace=True)
        object_evolutions["ocim:from"] = object_evolutions["ocel:timestamp"]
        object_evolutions.drop("ocel:timestamp", axis=1, inplace=True)
        object_evolutions.drop("ocel:type", axis=1, inplace=True)
        object_evolutions["ocim:to"] = pd.to_datetime(maxtime)

        object_evolutions.sort_values(["ocel:oid", "ocim:from"], inplace=True)
        prev_oid = None
        prev_index = None
        not_attribute_columns = ["ocel:oid", "ocel:type", "ocel:field", "ocim:from", "ocim:to"]
        for index, row in object_evolutions.iterrows():
            oid = row["ocel:oid"]
            if oid == prev_oid:
                field = row["ocel:field"]
                updated_value = row[field]
                time = row["ocim:from"]
                object_evolutions.at[prev_index, "ocim:to"] = time
                for col in object_evolutions.columns:
                    if col not in not_attribute_columns and col != field:
                        object_evolutions.at[index, col] = object_evolutions.at[prev_index, col]
            prev_oid = oid
            prev_index = index
        object_evolutions.drop("ocel:field", axis=1, inplace=True)
        self.__object_evolutions_frame = object_evolutions

    def __create_attribute_enriched_event_frame(self):
        event_frame = self.__event_frame
        e2o_frame = self.__e2o_frame
        object_evolution_frame = self.__object_evolutions_frame
        event_objects = event_frame.merge(e2o_frame, on = "ocel:eid")
        temp = event_objects.merge(object_evolution_frame, on = "ocel:oid")
        event_objects_attributes_frame = temp[(temp['ocel:timestamp'] >= temp['ocim:from']) & (temp['ocel:timestamp'] < temp['ocim:to'])]
        event_objects_attributes_frame = event_objects_attributes_frame.drop(["ocim:from", "ocim:to"], axis=1)
        event_objects_attributes_frame.rename(columns={
            attr: "ocim:attr:" + attr
            for attr in event_objects_attributes_frame.columns
            if not attr.startswith("ocel:")
        }, inplace=True)
        self.__event_objects_attributes_frame = event_objects_attributes_frame


    def get_event_object_attributes_frame(self):
        return self.__event_objects_attributes_frame

    def get_o2o_frame(self):
        return self.__o2o_frame

    def get_objects_frame(self):
        return self.__object_frame

    def get_event_frame(self):
        return self.__event_frame

    def get_e2o_frame(self):
        return self.__e2o_frame

    def get_object_evolutions_frame(self):
        return self.__object_evolutions_frame