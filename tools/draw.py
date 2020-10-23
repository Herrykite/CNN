import matplotlib.pyplot as plt
from ConvNet.config.defaults import get_cfg_defaults

cfg = get_cfg_defaults()


def draw_train_process(title, i, loss, label):
    plt.title(title, fontsize=cfg.VISUAL.TITLE_FRONT_SIZE)
    plt.xlabel(cfg.VISUAL.X_LABEL, fontsize=cfg.VISUAL.LABEL_FRONT_SIZE)
    plt.plot(i, loss, color=cfg.VISUAL.LINE_COLOR, label=label)
    plt.legend()
    plt.grid()
    plt.show()
