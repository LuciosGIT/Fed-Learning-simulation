from typing import Iterable
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAdagrad
from flwr.app import ArrayRecord, ConfigRecord, Message


class CustomFedAdagrad(FedAdagrad):
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training and maybe do LR and epsilon decay."""
        # Diminui o learning rate a cada 5 rodadas
        if server_round % 5 == 0 and server_round > 0:
            config["lr"] *= 0.5
            print("LR decreased to:", config["lr"])
        # Atualiza epsilon conforme desejado
        if "epsilon" in config:
            # Exemplo: dobra epsilon a cada 10 rodadas (ajuste sua lógica aqui)
            if server_round % 10 == 0 and server_round > 0:
                config["epsilon"] *= 2
                print("Epsilon increased to:", config["epsilon"])
        # Passa o restante para a classe-pai
        return super().configure_train(server_round, arrays, config, grid)
