import torch
import tensorrt as trt

from polygraphy.logger import G_LOGGER
G_LOGGER.module_severity = G_LOGGER.INFO
G_LOGGER.severity = G_LOGGER.INFO

print("Logger set up!")