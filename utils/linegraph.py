import matplotlib.pyplot as plt
import numpy as np
import os


def linegraph_step(
    model,
    target_name, 
    given_day, 
    end_day,
    target_is_next, 
    MAE, 
    MSE, 
    RMSE, 
    underestimated,
    overestimated,
    max_error,
):
    
    input_len = len(MAE)
    
    title = f"plot_{model}_{target_name}{end_day}_day-{given_day}to{given_day+input_len-1}_target_is_next-{target_is_next}"
    
    path_plot = os.path.join('./plots/', title + '.png')
    
    fig, ax = plt.subplots(figsize=(16, 11))
    ax1 = ax.twinx()
    
    ax.plot(np.arange(given_day, given_day + input_len, 1), MAE, marker='+', color='green', label='MAE')
    ax.plot(np.arange(given_day, given_day + input_len, 1), MSE, marker='x', color='purple', label='MSE')
    ax.plot(np.arange(given_day, given_day + input_len, 1), RMSE, marker='x', color='orange', label='RMSE')
    
    ax1.plot(np.arange(given_day, given_day + input_len, 1), underestimated, marker='v', color='blue', linestyle='dashed', label='underestimated')
    ax1.plot(np.arange(given_day, given_day + input_len, 1), overestimated, marker='^', color='red', linestyle='dashed', label='overestimated')
    
    ax.plot(np.arange(given_day, given_day + input_len, 1), max_error, marker='o', color='black', label='max error')

    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Error')
    ax1.set_ylabel('under or over estimated ratio')
    plt.title(title)
    plt.xticks(np.arange(given_day, given_day + input_len, 1))
    ax.legend(loc=2)
    ax1.legend(loc=0)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(path_plot)
    plt.show()