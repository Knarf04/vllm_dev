# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import TYPE_CHECKING, Callable

import vllm.envs as envs
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBaseType
from vllm.logger import init_logger

from .base import KVConnectorBase

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class KVConnectorFactory:
    _registry: dict[str, Callable[[], type[KVConnectorBaseType]]] = {}

    @classmethod
    def register_connector(cls, name: str, module_path: str,
                           class_name: str) -> None:
        """Register a connector with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")

        def loader() -> type[KVConnectorBaseType]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def create_connector(cls, rank: int, local_rank: int,
                            config: "VllmConfig") -> KVConnectorBase:
        connector_name = config.kv_transfer_config.kv_connector
        if connector_name not in cls._registry:
            raise ValueError(f"Unsupported connector type: {connector_name}")

        connector_cls = cls._registry[connector_name]()
        assert issubclass(connector_cls, KVConnectorBase)
        return connector_cls(rank, local_rank, config)

# Register various connectors here.
# The registration should not be done in each individual file, as we want to
# only load the files corresponding to the current connector.
KVConnectorFactory.register_connector(
    "PyNcclConnector",
    "vllm.distributed.kv_transfer.kv_connector.simple_connector",
    "SimpleConnector")

KVConnectorFactory.register_connector(
    "MooncakeConnector",
    "vllm.distributed.kv_transfer.kv_connector.simple_connector",
    "SimpleConnector")

KVConnectorFactory.register_connector(
    "LMCacheConnector",
    "vllm.distributed.kv_transfer.kv_connector.lmcache_connector",
    "LMCacheConnector")

KVConnectorFactory.register_connector(
    "MooncakeStoreConnector",
    "vllm.distributed.kv_transfer.kv_connector.mooncake_store_connector",
    "MooncakeStoreConnector")