from enum import Enum

from notneat_py.graph import generate_neural_network


class TradeState(Enum):
    ENTERING = 1
    EXITING = 2


class TradeGenome:
    def __init__(self):
        self.trade_state = TradeState.ENTERING
        # Inputs: last 50 hours of ohlc data (normalized), current weekday, hour, minute
        # Outputs: buy limit price, buy stop price, sell limit price, sell stop price
        """
        Usage: Trade entry neural network reads raw (non-normalized) price inputs and generates buy and sell stop and limit order prices.
        Order prices will change as inputs change, until price crosses one of the orders at which point the trade will switch from ENTERING to EXITING state.
        """
        self.entry_network = generate_neural_network(layers=[5 * 10 + 3, 60, 60, 10, 4])
        self.exit_network = generate_neural_network(layers=[5 * 10 + 3, 60, 60, 10, 4])

