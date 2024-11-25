import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os, sys, time
import logging
from datetime import datetime

clear = lambda: os.system("clear")
clear()

sys.path.append(os.path.abspath("../"))
current = os.path.dirname(os.path.realpath(__file__))
# Logger para crear archivo de logs y prints
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=current + "/SSD_inventario.log",
    encoding="utf-8",
    filemode="w",
    format="%(levelname)s:%(message)s",
    # level=logging.DEBUG,
    level=logging.WARNING,
)
logging.getLogger("matplotlib.font_manager").disabled = True

today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logging.warning(today)


class InventorySystem:
    """Double product inventory discrete simulation with periodically reorder and
    order size policy."""

    def __init__(
        self,
        DEMAND_SIZES,
        DEMAND_PROB,
        client_satisfied,
        client_not_satisfied,
        product_price,
        LAMBDA_EXP,
        start_time,
        START_INVENTORY_P1,
        START_INVENTORY_P2,
        MAX_INVENTORY_P1,
        MAX_INVENTORY_P2,
        Holding_cost,
        Holding_total,
        t,
        t_event,
        t_0,
        t_shortage,
        order_prices,
        order_penalty,
        reorder_time,
        order_base_cost,
        MU_ORDER,
        SIGMA_ORDER,
        last_change,
        MAX_TIME,
    ):
        # initialize values
        self.DEMAND_SIZES = DEMAND_SIZES
        self.DEMAND_PROB = DEMAND_PROB
        self.Lambda = LAMBDA_EXP
        self.product_price = product_price
        self.start_time = start_time
        self.level_p1 = START_INVENTORY_P1
        self.level_p2 = START_INVENTORY_P2
        self.level_p1_max = MAX_INVENTORY_P1
        self.level_p2_max = MAX_INVENTORY_P2
        self.order_prices = order_prices
        self.order_penalty = order_penalty
        self.Holding_cost = Holding_cost
        self.Holding_total = Holding_total
        self.t = t
        self.t_event = t_event
        self.t_0 = t_0
        self.lead_time = 0
        self.t_order = []
        self.t_shortage = t_shortage
        self.client_satisfied = client_satisfied
        self.client_not_satisfied = client_not_satisfied
        self.order_base_cost = order_base_cost
        self.all_orders_cost = 0
        self.reorder_time = reorder_time
        self.last_change = last_change
        self.order_incoming_P1 = 0
        self.order_incoming_P2 = 0
        self.MU_ORDER = MU_ORDER
        self.SIGMA_ORDER = SIGMA_ORDER
        self.history = []

        self.T = MAX_TIME

    # Imprimir un objeto de esta clase:
    def __repr__(self):
        this_Inventory = pd.DataFrame(
            {
                "Variables": [
                    "price p_1",
                    "price p_2",
                    "P_1_max",
                    "P_2_max",
                    "P_1_init",
                    "P_2_init",
                    "Llegadas",
                    "Demand size",
                    "P_1 prob()",
                    "P_2 prob()",
                    "Order_cost",
                    "Lead(mu)",
                    "Lead(sigma)",
                    "H_cost",
                    "Order_cost_p1",
                    "Order_cost_p2",
                    "Order_penalty",
                    "Tiempo simulado",
                ],
                "Valor": [
                    f"{self.product_price['prod1']} euros",
                    f"{self.product_price['prod2']} euros",
                    self.level_p1_max,
                    self.level_p2_max,
                    self.level_p1,
                    self.level_p2,
                    f"Poisson({self.Lambda})",
                    f"{self.DEMAND_SIZES}",
                    f"{self.DEMAND_PROB['prod1']}",
                    f"{self.DEMAND_PROB['prod2']}",
                    self.order_base_cost,
                    f"{self.MU_ORDER} hours",
                    f"{self.SIGMA_ORDER} hours",
                    f"{self.Holding_cost} *un *t",
                    self.order_prices["p1"],
                    self.order_prices["p2"],
                    f"{self.order_penalty['percentage']} %",
                    f"{self.T} horas",
                ],
            }
        )

        return f"Inventario:\n {this_Inventory}"

    def Log_Results(self):
        """Función para imprimir los resultados en el Log"""
        results = pd.DataFrame(
            {
                "Variables": [
                    "Holding",
                    "Satisfechos",
                    "No satisfechos",
                    "Tiempo en cero p1",
                    "Tiempo en cero p2",
                    "perdida en p1",
                    "perdida en p2",
                    "Beneficio",
                    "p1_Benefit",
                    "p2_Benefit",
                ],
                "Valor": [
                    self.Holding_total,
                    self.client_satisfied,
                    self.client_not_satisfied,
                    (
                        self.productos["prod1"]["sin_inventario"][
                            len(self.productos["prod1"]["sin_inventario"]) - 1
                        ]
                        - self.productos["prod1"]["sin_inventario"][0]
                    ),
                    (
                        self.productos["prod2"]["sin_inventario"][
                            len(self.productos["prod2"]["sin_inventario"]) - 1
                        ]
                        - self.productos["prod2"]["sin_inventario"][0]
                    ),
                    self.productos["prod1"]["perdidas"],
                    self.productos["prod2"]["perdidas"],
                    self.beneficio,
                    self.productos["prod1"]["total_benefit"][-1],
                    self.productos["prod2"]["total_benefit"][-1],
                ],
            }
        )
        return f"Resultados: \n {results.round(2)}"

    def run_process(self, display_chart):
        self.Lista = {
            "Tc": float("inf"),
            "Tp": float("inf"),
        }
        self.beneficio = 0
        self.productos = {
            "tiempo": [],
            "prod1": {
                "perdidas": 0,
                "sin_inventario": [],
                "total_benefit": [],
                "benefits": [],
                "loss_sales": [],
                "level": [],
            },
            "prod2": {
                "perdidas": 0,
                "sin_inventario": [],
                "total_benefit": [],
                "benefits": [],
                "loss_sales": [],
                "level": [],
            },
        }
        logging.warning(f"===== start run() {today}")
        self.productos["prod1"]["level"].append(70)
        self.productos["prod2"]["level"].append(70)
        self.productos["prod1"]["total_benefit"].append(0)
        self.productos["prod2"]["total_benefit"].append(0)
        self.productos["tiempo"].append(0)
        # Generate exponential (arrival of demand)
        Z = np.random.exponential(self.Lambda)
        if Z > self.T:
            logging.error("***Z>T****")
            return -1
        else:
            # Primera demanda de clientes, llegando en t=Z
            logging.debug(f"Primera demanda en t = {Z}")
            self.demands(Z)
            ##! ==========================================
            # logging.error("self.Lista_Tc = {}".format(self.Lista["Tc"]))
            # logging.error("self.Lista_Tp = {}".format(self.Lista["Tp"]))
            logging.debug("=======================================")
            counter = 0  # Verificar el uso de esto
            last_order_time = 0  # Tiempo del último pedido periódico
            time_diff = 0
            while (self.Lista["Tp"] != float("inf")) or (
                self.Lista["Tc"] != float("inf")
            ):

                # Lógica de pedido periódico (cada 168 horas)
                time_diff = self.t - last_order_time
                logging.debug(
                    f"--> t={round(self.t,2)}, last={round(last_order_time,2)},dif = {round(time_diff,2)} *******"
                )
                if time_diff >= self.reorder_time:
                    # Tiempo en que se hace pedido:

                    logging.debug(
                        f"\n\n ** pedido en t = {self.t}, r ={self.reorder_time}"
                    )
                    last_order_time = self.t  # Reiniciar el contador
                    self.order_periodically(self.t, time_diff)
                    logging.debug(self.Lista["Tp"])
                    # self.t_event = self.t
                    # self.Lista["Tc"] = self.t
                    logging.debug(f"arrives: {self.t}")
                    logging.debug(f"Tp = {self.Lista['Tp']}")
                    logging.debug(f"Tc = {self.Lista['Tc']}")
                    self.t_order.append(self.Lista["Tc"])  # Guardar para gráficos
                    self.productos["tiempo"].append(self.Lista["Tc"])

                if self.Lista["Tc"] < self.Lista["Tp"]:
                    # logging.error(f"!!! Tp > Tc : {self.Lista['Tc']}")
                    self.t_event = self.Lista["Tc"]
                    # logging.warning(f"t: {self.t}")
                    self.Lista["Tc"] = float("inf")
                    # Rutina de llegada de cliente(t_event)
                    self.demands(self.t_event)

                if self.Lista["Tp"] < self.Lista["Tc"]:
                    logging.debug("\n\n if2: Tp < Tc")
                    self.productos["prod1"]["level"].append(
                        self.order_incoming_P1 + self.productos["prod1"]["level"][-1]
                    )
                    self.productos["prod2"]["level"].append(
                        self.order_incoming_P2 + self.productos["prod2"]["level"][-1]
                    )
                    # TODO pagar el pedido

                    self.t_event = self.Lista["Tp"]
                    self.Lista["Tp"] = float("inf")
                    # Rutina de llegada de pedido(t_event)
                    # self.order_routine(self.t_event)
                if self.t >= self.T:
                    logging.warning(
                        "*self.t >= self.T****** (T={}, t={}) t > T - counter = {}".format(
                            self.T, self.t, counter
                        )
                    )
                    return -1

            logging.warning(self.Log_Results())  # Imprime un df con resultados
            if display_chart:
                # self.test_graph()
                self.step_graph()
                # self.Indicators_graph()
        # Retorno de toda la Clase InventorySystem:
        return self.beneficio

    def update_benefit(self, prod_demand, prod_name):
        """Función para calcular el beneficio al generarse la demanda

        Parameters:
            prod_demand (int): Cantidad de producto demandado
            prod_name (String): Nombre del producto (prod1 o prod2)
        Returns
            None
        """
        # logging.debug(f"before benefit on {prod_name} {self.productos[prod_name]["total_benefit"]}, ({prod_demand})")
        self.productos[prod_name]["total_benefit"].append(
            self.productos[prod_name]["total_benefit"][-1]
            + prod_demand * self.product_price[prod_name]
        )
        self.productos[prod_name]["benefits"].append(
            prod_demand * self.product_price[prod_name]
        )
        # logging.debug(f"after benefit on {prod_name} {self.productos[prod_name]["total_benefit"]}, ({prod_demand})\n")
        self.beneficio = self.beneficio + (prod_demand * self.product_price[prod_name])

    def update_benefit_onShort(self, prod_demand, prod_name, prod_level):
        """Función para calcular el beneficio cuando el nivel de inventario no supera la demanda

        Parameters:
            prod_demand (int): Cantidad de producto demandado
            prod_name (String): Nombre del producto (prod1 o prod2)
            prod_level (int): Cantidad de producto en inventario
        Returns
            None
        """
        logging.info(f"update shortage  {prod_name}")
        logging.info(
            f"= {(prod_demand - prod_level) * self.product_price[prod_name]}, t = {self.t}"
        )
        logging.info(f"{prod_name},{prod_demand}, {prod_level}")
        # Contabilizar cuantos (cantidad y $) de cada prod se pierden
        self.beneficio = self.beneficio + (prod_level * self.product_price[prod_name])
        self.productos[prod_name]["total_benefit"].append(
            self.productos[prod_name]["total_benefit"][-1]
            + prod_level * self.product_price[prod_name]
        )
        self.productos[prod_name]["benefits"].append(
            prod_level * self.product_price[prod_name]
        )
        self.productos[prod_name]["perdidas"] = (
            self.productos[prod_name]["perdidas"]
            + (prod_demand - prod_level) * self.product_price[prod_name]
        )
        self.productos[prod_name]["loss_sales"].append(
            self.productos[prod_name]["perdidas"]
        )
        # Actualizar el nivel
        # self.update_level(prod_demand)
        self.productos[prod_name]["level"].append(int(0))

    def update_level(self, prod_demand, prod_name):
        """Función para actualizar el nivel de inventario

        Parameters:
            prod_demand (int): Cantidad de producto demandado
            prod_name (String): Nombre del producto (prod1 o prod2)
        Returns:
            None
        """

        logging.info("update level")
        self.productos[prod_name]["level"].append(
            int(self.productos[prod_name]["level"][-1] - prod_demand)
        )
        logging.info(f"nivel {prod_name}: {self.productos[prod_name]['level']}")
        # if prod_name == "prod1":
        #     self.level_p1 = self.level_p1 - prod_demand
        # if prod_name == "prod2":
        #     self.level_p2 = self.level_p2 - prod_demand

    def order_periodically(self, current_time, time_difference):
        """
        Función para procesar un pedido periódico al proveedor.
        Parameters:
            current_time (float): Tiempo actual de la simulación
            time_difference (float): Diferencia entre tiempo corriente y el tiempo de pedido
        Returns:
            None
        """
        # TODO Saber que el pedido se paga y se toma en cuenta al LLEGAR a bodega
        # TODO Calcular H
        self.Holding_total = self.Holding_total + (
            self.t - self.productos["tiempo"][-4:][0]
        ) * self.Holding_cost * (
            self.productos["prod1"]["level"][-1] + self.productos["prod2"]["level"][-1]
        )
        # Generar el tiempo de entrega (lead time)
        lead_time = np.random.normal(self.MU_ORDER, self.SIGMA_ORDER)
        time_correction = current_time - abs(time_difference - self.reorder_time)
        logging.info(f"t={self.t}, tc={time_correction}, lead={lead_time}")

        # Actualizar Tp
        self.Lista["Tp"] = time_correction + lead_time
        self.t = time_correction + lead_time
        logging.debug(f"self.t={self.t}")
        # Calcular cuántos productos pedir para reabastecer al máximo nivel
        # Si prod1 es menor o igual a 600 -> 1 euro la unidad, si se piden más de 600 el precio desciende a 75 céntimos.
        # Si prod2 es menor que 800 el precio es de 1.5 euros la unidad, mientras que si se piden más de 800 el precio desciende a 1.25 euro
        self.order_incoming_P1 = (
            self.level_p1_max - self.productos["prod1"]["level"][-1]
        )
        self.order_incoming_P2 = (
            self.level_p2_max - self.productos["prod2"]["level"][-1]
        )
        # Descuento por cantidad
        if self.order_incoming_P1 > self.order_prices["p1"]["limit"]:
            prod1_price = self.order_prices["p1"]["above"]
        else:
            prod1_price = self.order_prices["p1"]["under"]

        if self.order_incoming_P2 > self.order_prices["p2"]["limit"]:
            prod2_price = self.order_prices["p2"]["above"]
        else:
            prod2_price = self.order_prices["p2"]["under"]

        # Penalización por tiempo de envío
        # si el pedido llega con 3 horas de retraso en la entrega --> -0.03% del valor del pedido,
        if lead_time - self.MU_ORDER > 3:
            self.all_orders_cost += (
                self.order_base_cost
                + (self.order_incoming_P1 * prod1_price)
                + (self.order_incoming_P2 * prod2_price)
            ) * 0.97
        # si el pedido llega con 3 horas de adelanto en la entrega --> +0.03% del valor del pedido,
        if lead_time - self.MU_ORDER < -3:
            self.all_orders_cost += (
                self.order_base_cost
                + (self.order_incoming_P1 * prod1_price)
                + (self.order_incoming_P2 * prod2_price)
            ) * 1.03
        self.level_p1 = self.level_p1 + self.order_incoming_P1
        self.level_p2 = self.level_p2 + self.order_incoming_P2
        logging.debug(
            f"Pedido: ({self.order_incoming_P1}) - ({self.order_incoming_P2})"
        )
        # self.productos["tiempo"].append(time_correction + lead_time)
        self.history.append((current_time, self.level_p1 + self.level_p2))

    def order_routine(self, event_time):
        """Función para la rutina aperiódica de pedidos según nivel de inventario

        Args:
            event_time (float): Tiempo de la simulación
        """
        self.t = event_time
        logging.info("self.level={}".format(self.level_p1))
        logging.info("self.incoming={}".format(self.order_incoming_P1))
        # update ordering costs

        order_cost = self.order_base_cost + (
            self.order_incoming_P1 * self.order_prices["p1"]["under"]
        )
        self.all_orders_cost = self.all_orders_cost + order_cost
        self.level_p1 = self.level_p1 + self.order_incoming_P1

        self.order_incoming = 0
        # self.history.append((self.t, self.level_p1))
        self.productos["tiempo"].append(self.t)
        if self.t_shortage > 0:
            logging.info("self.t_shortage = {}".format(self.t_shortage))
            self.t_shortage = 0

    def demands(self, T_suc):
        """Función para generar las demandas de productos según probabilidad,
        Generar demandas y actualizar el costo de almacenamiento, actualizar el inventario,
        verificar el nivel del mismo, generar futuros tiempos de llegada.

        Parameters:
            T_suc (float): Tiempo de la simulación
        """
        # Update Holding cost
        # Cuando ocurre la demanda de productos, se está cobrando almacenaje de ambos productos
        # TODO: El Holding no se calcula bien por el resultado de T_suc - self.t despues de un PEDIDO
        bef = self.Holding_total
        if self.t > T_suc:
            self.Holding_total = self.Holding_total + (
                T_suc - self.productos["tiempo"][-4:][0]
            ) * self.Holding_cost * (
                self.productos["prod1"]["level"][-1]
                + self.productos["prod2"]["level"][-1]
            )
        else:
            self.Holding_total = self.Holding_total + (
                T_suc - self.t
            ) * self.Holding_cost * (
                self.productos["prod1"]["level"][-1]
                + self.productos["prod2"]["level"][-1]
            )
        self.t = T_suc  # Update t
        # Generar demanda de pedidos P1 y P2
        demandas = {
            "prod1": np.random.choice(self.DEMAND_SIZES, 1, self.DEMAND_PROB["prod1"])[
                0
            ],
            "prod2": np.random.choice(self.DEMAND_SIZES, 1, self.DEMAND_PROB["prod2"])[
                0
            ],
        }
        # TODO: Verificar si cada cliente siempre se lleva ambos productos
        for producto, demanda in demandas.items():
            # logging.error(f"*** nivel {producto}: {self.productos[producto]["level"][-1]}")
            # nivel_actual_raw = getattr(self, f"level_p{producto[-1]}")
            # demanda = demanda_raw
            # nivel_actual = int(nivel_actual_raw)
            nivel_actual = self.productos[producto]["level"][-1]

            # logging.error(f"demanda ({demanda}), level_{producto} ({nivel_actual})")
            if demanda <= nivel_actual:
                self.update_benefit(demanda, producto)
                self.update_level(demanda, producto)
                self.client_satisfied += 1
            else:
                self.update_benefit_onShort(demanda, producto, nivel_actual)
                self.client_not_satisfied += 1
                self.level_p1 = 0
                # Si la demanda esta por encima del nivel de inventario, se despacha lo restante y se pierde lo que no se puede suplir al cliente.
                if nivel_actual == 0:
                    # logging.error(f"sin inventario en {producto}")
                    self.t_shortage = self.t
                    self.productos[producto]["sin_inventario"].append(self.t)
            # Generar siguiente llegada:
            new_arrival_time = np.random.exponential(self.Lambda)

            if (self.t + new_arrival_time) < self.T:
                self.Lista["Tc"] = self.t + new_arrival_time
                # logging.warning(f"dmd -> Tc = {self.Lista['Tc']}")
                # self.history.append(
                #     (self.t + new_arrival_time, (self.level_p1 + self.level_p2))
                # )
        self.productos["tiempo"].append(self.t + new_arrival_time)

    def step_graph(self):
        """Displays a step line chart of inventory level"""
        # Eilminar el tiempo de pedidos en t y graficar H

        #         "sin_inventario": [],
        #         "total_benefit": [],
        #         "loss_sales": [],
        #         "benefits": [],
        #         "level": [],
        ##? create subplot
        fig, axs = plt.subplots(2, figsize=(10, 7))
        axs[0].grid(
            which="major", alpha=0.4, color="gray", linestyle="-", linewidth=0.3
        )
        axs[1].grid(
            which="major", alpha=0.4, color="gray", linestyle="-", linewidth=0.3
        )
        ##? plot simulation data
        # axs[0].plot(
        #     self.productos["tiempo"],
        #     self.productos["prod1"]["level"],
        #     "o--",
        #     # markersize=0.5,
        #     color="gray",
        #     alpha=0.5,
        # )
        axs[0].step(
            self.productos["tiempo"],
            self.productos["prod1"]["level"],
            where="post",
            label="Unidades de producto 1",
        )
        axs[1].step(
            self.productos["tiempo"],
            self.productos["prod2"]["level"],
            where="post",
            label="Unidades de producto 2",
        )
        ##? Agregar flechas para señalar los tiempos de orden
        for ot in self.t_order:
            if ot != 0:
                axs[0].axvline(
                    x=ot,
                    color="dodgerblue",
                    linestyle="--",
                    label=f"Pedido en {round(ot,1)} h",
                )
                axs[0].text(
                    x=ot - (ot / 100),
                    y=200,
                    s=f"t={round(ot,2)} h",
                    rotation="vertical",
                    fontsize="x-small",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        edgecolor="dodgerblue",
                        facecolor="white",
                        alpha=0.8,
                    ),
                    ha="left",
                )
                axs[1].axvline(
                    x=ot,
                    color="dodgerblue",
                    linestyle="--",
                    label=f"Pedido en {round(ot,1)} h",
                )
        # ? titles and legends
        axs[0].set_title(f" Simulación con t = {int(self.T/24)} días")
        axs[0].set_ylabel("Unidades de producto 1")
        axs[1].set_ylabel("Unidades de producto 2")
        axs[1].set_xlabel("Horas")
        plt.savefig(f"Inventario con {int(self.T/24)} días")
        plt.show()

    def Indicators_graph(self):
        # Eliminar los tiempos de pedido (array total_benefit no los registra)
        for i in self.t_order:
            logging.error(i)
            self.productos["tiempo"].remove(i)
        ##? create subplot
        fig, axs = plt.subplots(2, figsize=(10, 7))
        axs[0].grid(
            which="major", alpha=0.4, color="gray", linestyle="-", linewidth=0.3
        )
        axs[1].grid(
            which="major", alpha=0.4, color="gray", linestyle="-", linewidth=0.3
        )
        axs[0].step(
            self.productos["tiempo"],
            self.productos["prod1"]["total_benefit"],
            where="post",
        )
        axs[1].step(
            self.productos["tiempo"],
            self.productos["prod2"]["total_benefit"],
            where="post",
        )
        axs[0].set_title(f" Simulación con t = {int(self.T/24)} días")
        axs[0].set_ylabel("Beneficio de producto 1")
        axs[1].set_ylabel("Beneficio de producto 2")
        axs[1].set_xlabel("Horas")
        plt.show()

    def test_graph(self):
        fig = plt.figure(figsize=(14, 7))
        # ! Test printing
        plt.grid(which="major", alpha=0.4, color="gray", linestyle="-", linewidth=0.3)
        plt.plot(
            self.productos["tiempo"],
            self.productos["prod1"]["level"],
            "o--",
            color="gray",
            alpha=0.95,
        )

        plt.step(
            self.productos["tiempo"],
            self.productos["prod1"]["level"],
            where="post",
            label="Unidades de producto 1",
        )
        for ot in self.t_order:
            if ot != 0:
                plt.axvline(
                    x=ot,
                    color="dodgerblue",
                    linestyle="--",
                    label=f"Pedido en {round(ot,1)} h",
                )
                plt.text(
                    x=ot - (ot / 100),
                    y=300,
                    s=f"t={round(ot,2)} h",
                    rotation="vertical",
                    fontsize="small",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        edgecolor="dodgerblue",
                        facecolor="white",
                        alpha=0.8,
                    ),
                    ha="left",
                )
        plt.show()


def main():
    # Instance of InventorySystem object
    inventory_process = InventorySystem(
        start_time=time.time(),
        ## !CLIENTES
        DEMAND_SIZES=[1, 2, 3, 4],  # possible customer demand sizes
        product_price={"prod1": 2.5, "prod2": 3.5},
        DEMAND_PROB={
            "prod1": [0.3, 0.4, 0.2, 0.1],
            "prod2": [0.2, 0.2, 0.4, 0.2],
        },  # probability of each demand size
        client_satisfied=0,
        client_not_satisfied=0,
        ## !TIEMPOS
        LAMBDA_EXP=1.5,  # average time between customer demands (months)
        t=0,
        t_event=0,
        t_0=0,
        t_shortage=0,
        MAX_TIME=24 * 30 * 2,  # n Meses en horas
        ## !PEDIDOS
        # reorder_point=500,  # Cambiar a periódico
        reorder_time=168,  # Cada semana
        order_base_cost=100,
        MU_ORDER=48,  # Lead promedio
        SIGMA_ORDER=3.5,  # Lead desv. estándar
        order_prices={
            "p1": {"under": 1, "above": 0.75, "limit": 600},
            "p2": {"under": 1.5, "above": 1.25, "limit": 800},
        },
        order_penalty={"percentage": 0.03, "time_base": 48},
        # si el pedido llega con 3 horas de retraso en la entrega --> -0.03% del valor del pedido,
        # si el pedido llega con 3 horas de adelanto en la entrega --> +0.03% del valor del pedido,
        ## !ALMACENAMIENTOS
        Holding_cost=0.0002,
        Holding_total=0,  # Starting value
        last_change=0.0,
        START_INVENTORY_P1=70,
        START_INVENTORY_P2=70,
        MAX_INVENTORY_P1=1000,
        MAX_INVENTORY_P2=1500,
    )
    logging.warning(inventory_process)  # Print Inventory System object
    test = []
    # TODO Correr para varios valores de Pedido Máximo e Inventario inicial
    for i in [1]:
        test.append(
            inventory_process.run_process(
                display_chart=True,
                # display_chart=False,
            )
        )
        logging.error(test)
    # inventory_process.Indicators_graph()


# run simulation
if __name__ == "__main__":

    # Simular el comportamiento de almacén durante 5 meses para estimar
    # el beneficio esperado, la proporción de clientes cuya demanda se satisface
    # completamente y el porcentaje de tiempo que el nivel del inventario permanece a cero.
    # Para ello, supondremos que el nivel del inventario inicial es de 70 unidades de ambos productos
    start_t = time.time()
    main()
    end_t = time.time()
    logging.warning("time: {:,} s".format(round(end_t - start_t, 4)))

# [0, 168, 336, 504, 672, 840, 1008, 1176, 1344, 1512, 1680, 1848, 2016, 2184, 2352, 2520, 2688, 2856, 3024, 3192, 3360, 3528]
