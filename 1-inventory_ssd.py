import numpy as np
import matplotlib.pyplot as plt
import os, sys, time
import logging
from datetime import datetime

clear = lambda: os.system("clear")
# clear()
sys.path.append(os.path.abspath("../"))
current = os.path.dirname(os.path.realpath(__file__))

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=current + "/SSD_inventory.log",
    encoding="utf-8",
    filemode="w",
    format="%(levelname)s:%(message)s",
    # level=logging.DEBUG,
    level=logging.WARNING,
)
logging.getLogger("matplotlib.font_manager").disabled = True

today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logging.debug(today)


class InventorySystem:
    """Single product inventory system using a fixed reorder point
    and order size policy. Inventory is reviewed at regular intervals"""

    def __init__(
        self,
        DEMAND_SIZES,
        DEMAND_PROB,
        LAMBDA_EXP,
        start_time,
        START_INVENTORY,
        shortage_cost,
        holding_cost,
        Holding_total,
        order_cost,
        t,
        t_event,
        t_0,
        t_shortage,
        client_sat,
        client_not_sat,
        reorder_point,
        mu_order,
        order_item_cost,
        costo_unitario,
        last_change,
    ):
        # initialize values
        self.DEMAND_SIZES = DEMAND_SIZES
        self.DEMAND_PROB = DEMAND_PROB
        self.Lambda = LAMBDA_EXP
        self.start_time = start_time
        self.level = START_INVENTORY
        self.shortage_cost = shortage_cost
        self.holding_cost = holding_cost
        self.Holding_total = Holding_total
        self.t = t
        self.t_event = t_event
        self.t_0 = t_0
        self.time_arrive_order = 0
        self.t_shortage = t_shortage
        self.client_sat = client_sat
        self.client_not_sat = client_not_sat
        self.order_service_cost = 10
        self.order_amount = 0
        self.order_cost = order_cost
        self.all_orders_cost = 0
        self.reorder_point = reorder_point
        self.order_item_cost = order_item_cost
        self.last_change = last_change
        self.costo_unitario = costo_unitario
        self.order_incoming = 0
        self.P_capacity = START_INVENTORY
        self.mu_order = mu_order
        self.history = [(0, self.level)]
        # self.Lista_Tc = -1
        # self.Lista_Tp = -1
        self.tiempo_en_cero = 0
        self.Lista = {
            "Tc": float("inf"),
            "Tp": float("inf"),
        }
        self.beneficio = 0
        self.T = 24 * 30 * 1  # 5 Meses en horas

    def run(self, reorder_point: float, order_size: float, display_chart=False):
        logging.warning("===== =====       run()")
        # Generate exponential (arrival of demand)
        Z = np.random.exponential(self.Lambda)
        logging.info("\n self.arrival_time = {}".format(round(Z, 4)))
        if Z > self.T:
            logging.error("***Z>T****")
            return -1
        else:
            self.demands(Z, self.start_time)
            ##! ==========================================
            logging.info("self.Lista_Tc = {}".format(self.Lista["Tc"]))
            logging.info("self.Lista_Tp = {}".format(self.Lista["Tp"]))
            logging.info("================== END ========================")
            counter = 0
            # logging.info(self.Lista_Tp[len(self.Lista_Tp) - 1])
            while (self.Lista["Tp"] != float("inf")) or (
                self.Lista["Tc"] != float("inf")
            ):
                logging.info(
                    "*******  While T ={}, t={}  *******".format(self.T, self.t)
                )
                logging.info("self.Lista_Tc = {}".format(self.Lista["Tc"]))
                logging.info("self.Lista_Tp = {}".format(self.Lista["Tp"]))
                if self.Lista["Tc"] < self.Lista["Tp"]:
                    counter += 1
                    logging.info("     ==== demanda i = {} ====".format(counter))
                    self.t_event = self.Lista["Tc"]
                    self.Lista["Tc"] = float("inf")
                    # Rutina de llegada de cliente(t_event)
                    self.demands(self.t_event, self.start_time)

                if self.Lista["Tp"] < self.Lista["Tc"]:
                    #   if self.Lista['Tp'][len(self.Lista['Tp'])-1]<self.Lista_Tc[len(self.Lista_Tc)-1]:
                    counter += 1
                    self.t_event = self.Lista["Tp"]
                    self.Lista["Tp"] = float("inf")
                    # Rutina de llegada de pedido(t_event)
                    self.order_routine(self.t_event)
                if self.t >= self.T:
                    logging.warning(
                        "******* (T={}, t={}) t > T - counter = {}".format(
                            self.T, self.t, counter
                        )
                    )
                    logging.warning(
                        {
                            "beneficio": float(round(self.beneficio, 6)),
                            "holding_cost": float(round(self.Holding_total, 4)),
                            "level": self.level,
                            "reorder_point": round(self.reorder_point, 4),
                        }
                    )
                    logging.warning(
                        {
                            "t": round(self.t, 4),
                            "incoming": self.order_incoming,
                            "tiempo_shortage": self.tiempo_en_cero,
                            "c_not_sat": self.client_not_sat,
                            "c_sat": self.client_sat,
                            # "history":self.history
                        }
                    )
                    # logging.warning("self.history={}".format(self.history))
                    self.step_graph(self.reorder_point, self.history)
                    return -1

                if counter > 3420:
                    logging.warning(
                        "******* (T={}, t={}) salida - counter = {}".format(
                            self.T, self.t, counter
                        )
                    )
                    logging.info("salida del programa")
                    logging.info(
                        {
                            "beneficio": float(round(self.beneficio, 6)),
                            "holding_cost": float(round(self.Holding_total, 4)),
                            "level": self.level,
                            "reorder_point": round(self.reorder_point, 4),
                            "t": round(self.t, 4),
                            "incoming": self.order_incoming,
                            # "history":self.history
                        }
                    )
                    # step_graph(self.history)
                    self.step_graph(self.reorder_point, self.history)
                    return -1

    def demands(self, T_suc, start_time):
        # generate next demand size and time
        logging.info(
            "\n============   demand - level={}    ============".format(self.level)
        )
        logging.info(
            {
                "t": self.t,
                "arrival_time": T_suc,
                "actual_time": time.time() - start_time,
            }
        )
        logging.info(
            {
                "holding": round(self.Holding_total, 4),
                "T_suc": round(T_suc, 4),
                "time": round(time.time() - start_time, 4),
            }
        )
        logging.info(
            "holding = {}".format(
                (T_suc - (time.time() - start_time)) * self.holding_cost * self.level
            )
        )
        self.Holding_total = (
            self.Holding_total
            + (T_suc - (time.time() - start_time)) * self.holding_cost * self.level
        )

        self.t = T_suc
        # Generar demanda de pedidos
        demanda = np.random.choice(self.DEMAND_SIZES, 1, self.DEMAND_PROB)
        logging.info("demanda = {}".format(demanda[0]))

        if demanda[0] <= self.level:
            logging.info(
                "demanda ({}) <= self.level ({})".format(demanda[0], self.level)
            )
            self.beneficio = self.beneficio + demanda[0] * self.costo_unitario
            self.level = self.level - demanda[0]
            self.client_sat += 1

            if self.level == 0:
                # logging.error('sin inventario')
                self.tiempo_en_cero = self.t
                self.t_shortage = self.t
        else:
            logging.info(
                "!!!!!!!!!!!!! level={}, demand={}\n".format(self.level, demanda)
            )
            self.beneficio = self.beneficio + self.level * self.costo_unitario
            self.client_not_sat += 1
            logging.info("self.t_shortage={}".format(self.t_shortage))
            if self.t_shortage == 0:
                self.t_shortage = self.t
        logging.info("\n+++++++++++++++")
        # logging.info(self.level, self.reorder_point, self.order_incoming)
        if self.level <= self.reorder_point and self.order_incoming == 0:
            logging.info(
                "   LEVEL ({}) < REORDER_point ({})".format(
                    self.level, self.reorder_point
                )
            )
            logging.warning(
                "self.P_capacity={},self.level={}, order={}".format(
                    self.P_capacity, self.level, (self.P_capacity - self.level)
                )
            )
            self.order_incoming = (self.P_capacity - self.level) + (
                self.P_capacity - self.level + 30
            ) / 2
            # Tiempo lider (L) mu=48 horas, sigma=0.8
            logging.info("self.order_incoming = {}".format(self.order_incoming))
            self.time_arrive_order = np.random.normal(self.mu_order, 0.8)  # (L)
            self.Lista["Tp"] = (
                self.t + self.time_arrive_order
            )  # Verificar este tiempo q se dispara el valor de t

        logging.info(
            {
                "beneficio": float(round(self.beneficio, 6)),
                "holding_cost": float(round(self.Holding_total, 4)),
                "level": self.level,
                "reorder_point": round(self.reorder_point, 4),
                "t": round(self.t, 4),
                "arrival_time": round(T_suc, 4),
                "actual_time": round(time.time() - start_time, 4),
            }
        )
        # Generar siguiente llegada:
        new_arrival_time = np.random.exponential(self.Lambda)
        logging.info(
            {
                "new_arrival_time": round(new_arrival_time, 4),
                "t": round(self.t, 4),
                "pedido": round(self.time_arrive_order, 4),
                "arrival_time": round(T_suc, 4),
                "actual_time": round(time.time() - start_time, 4),
            }
        )
        if (self.t + new_arrival_time) < self.T:
            self.Lista["Tc"] = self.t + new_arrival_time
            self.history.append((self.t + new_arrival_time, self.level))

        logging.info(
            "self.Lista['Tc'] = {}, self.Lista['Tp'] = {}".format(
                self.Lista["Tc"], self.Lista["Tp"]
            )
        )

    def order_routine(self, event_time):
        logging.info(
            "\n===\n===============  here in order routine  =================="
        )
        self.Holding_total = (
            self.Holding_total
            + (event_time - (time.time() - event_time)) * self.holding_cost * self.level
        )
        self.t = event_time
        logging.info("self.level={}".format(self.level))
        logging.info("self.incoming={}".format(self.order_incoming))
        # update ordering costs
        self.order_cost = self.order_service_cost + (
            self.order_incoming * self.order_item_cost
        )
        self.all_orders_cost = self.all_orders_cost + self.order_cost
        self.level = self.level + self.order_incoming

        self.order_incoming = 0
        self.history.append((self.t, self.level))
        if self.t_shortage > 0:
            # self.history.append((self.t + self.t_shortage, self.level))
            logging.info("self.t_shortage = {}".format(self.t_shortage))
            self.t_shortage = 0

    def step_graph(self, reorder_point, history):
        """Displays a step line chart of inventory level"""
        # create subplot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.grid(which="major", alpha=0.4)
        # plot simulation data
        x_val = [x[0] for x in history]
        y_val = [x[1] for x in history]
        plt.step(x_val, y_val, where="post", label="Units in inventory")
        # plt.plot(x_val, 30, where = 'post', label='Units in inventory')
        plt.axhline(
            y=reorder_point, color="green", linestyle="--", label="Shortage threshold"
        )
        plt.axhline(y=0, color="red", linestyle="-", label="Shortage threshold")
        # plt.axhline(
        # y=self.order, color="green", linestyle="--", label="Reorder point"
        # )  # reorder point line
        # titles and legends
        plt.xlabel("Horas")
        plt.ylabel("Unidades en inventario")
        # plt.title(
        # f"Simulation output for system ({inventory.reorder_point}, {inventory.order_size})"
        # )
        plt.gca().legend()
        plt.show()


def main():
    clear()
    # simulation constant
    START_INVENTORY = 70  # units in inventory at simulation start P_0
    COST_ORDER_SETUP = 10.0  # fixed cost of placing an order
    COST_ORDER_PER_ITEM = 50.0  # variable cost of ordering an item
    COST_BACKLOG_PER_ITEM = 5.0  # monthly cost for each item in backlog
    COST_HOLDING_PER_ITEM = 1.0  # monthly cost for each item in inventory
    T = 12.0  # duration of the simulation (in months)
    inventory_process = InventorySystem(
        start_time=time.time(),
        # initialize values
        DEMAND_SIZES=[1, 2, 3, 4],  # possible customer demand sizes
        DEMAND_PROB=[0.1, 0.1, 0.4, 0.4],  # probability of each demand size
        LAMBDA_EXP=0.5,  # average time between customer demands (months)
        shortage_cost=0.0,
        holding_cost=0.0001,
        Holding_total=0,
        t=0,
        t_event=0,
        t_0=0,
        t_shortage=0,
        client_sat=0,
        client_not_sat=0,
        reorder_point=30,
        last_change=0.0,
        costo_unitario=2,
        order_item_cost=2,
        order_cost=10,
        mu_order=16,
        START_INVENTORY=50,
    )
    inventory_process.run(25, 40, display_chart=False)


# run simulation
if __name__ == "__main__":
    main()
    days = [r for r in range(1,26)]
    logging.warning(days)
    # for key, value in simulation.items():
    #     logging.info(f"{key}: {value}")
