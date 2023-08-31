import jax.numpy as jnp
import torch
import json

from script.model import GPModel
from script.GP_optimizer import BayOptimizer
from test_functions import simple_1d, func_wrapper, SyntheticFunction

from matplotlib import pyplot as plt
import matplotlib

cols = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]

def plot_GPs(model, x_train, y_train, benchmark, iteration, batch, batch_size=1, steps=1000):
    """
    plot a series of Gaussian process
    """
    assert(isinstance(model, BayOptimizer)), "`model` must be an instance of BayOptimizer"
    assert(isinstance(benchmark, SyntheticFunction))

    # data for visualization
    x_test = jnp.linspace(*benchmark.bounds, steps)
    y_test = benchmark.f(x_test)

    x_pred_mean = jnp.zeros((steps, ))
    x_pred_std = jnp.zeros(steps, )

    ## Prediction (just for visualization purpose)
    for i, sm in enumerate(model.SMs):
        pred_mean, pred_std = sm.inference(x_test)
        
        
        x_pred_mean += pred_mean
        x_pred_std += pred_std
        #acqs_test = pred_mean + model.j * pred_std
        #acqs_test = jnp.array([model.acqs_avg(x, model.j) for x in x_test])  # slow

        # visualization
        fig, ax = plt.subplots()
        fig.set_dpi(300)
        ax.scatter(x_train, y_train, marker='o', label="Observations", color=cols[0], alpha=0.5)
        ax.plot(x_test, y_test, label="Latent function", color=cols[0], linewidth=2)
        ax.plot(x_test, pred_mean, label="Predictive mean", color=cols[1])
        ax.fill_between(x_test.squeeze(), pred_mean-2*pred_std, pred_mean+2*pred_std, label="$2\sigma$", alpha=0.5, color=cols[1])
        #ax.scatter(x_new, y_new, marker='X', label="New point", color='r')
        #ax.plot(x_test, acqs_test, label="Acquisition function", color=cols[2])
        ax.legend()
        #plt.show()
        if batch_size > 1:
            plt.title(f"Iteration: {iteration+1} Batch: {batch+1} GP: {i+1}")
            fig_name = f"../results/j-ATS-UCB/1D/0/GP/Iter_{iteration+1}_Batch_{batch+1}_GP_{i+1}.svg"
        else:
            plt.title(f"Iteration: {iteration+1} GP: {i+1}")
            fig_name = f"../results/j-ATS-UCB/1D/0/GP/Iter_{iteration+1}_GP_{i+1}.svg"
        fig.savefig(fig_name)
        plt.close()

    acqs_test = (x_pred_mean + model.j * x_pred_std)/len(model.SMs)

    file_path = f"../data/j-ATS-UCB/1D/0/acqs_func/simple_4_iteration_{iteration}_batch_{batch}.json"
    with open(file_path, 'w') as f:
        json.dump(acqs_test.tolist(), f)
    return acqs_test

def plot_acqs_funcs(model, acqs_tests, x_train, y_train, x_news, y_news, benchmark, iteration, batch_size, steps=1000):
    assert(isinstance(model, BayOptimizer)), "`model` must be an instance of BayOptimizer"
    assert(isinstance(benchmark, SyntheticFunction))
    assert(batch_size > 1)

    # data for visualization
    x_test = jnp.linspace(*benchmark.bounds, steps)
    y_test = benchmark.f(x_test)

    acqs_tests = jnp.array(acqs_tests)

    fig, ax = plt.subplots()
    fig.set_dpi(300)
    ax.scatter(x_train, y_train, marker='o', label="Observations", color=cols[0], alpha=0.5)
    ax.plot(x_test, y_test, label="Latent function", color=cols[0], linewidth=2)
    for i in range(batch_size):
        ax.scatter(x_news[i], y_news[i], marker='X', label=f"new point {i+1}", color=cols[i+1])
        ax.plot(x_test, acqs_tests[i, :], label=f"acquisition function {i+1}", color=cols[i+1], linewidth=2)
    ax.legend()
    #plt.title(f"Acquisition functions of iteration: {iteration+1}")
    fig_name = f"../results/j-ATS-UCB/1D/0/acqs_func/Iter_{iteration+1}.svg"
    fig.savefig(fig_name)
    plt.close()

def plot_results_1D(x_train, y_train, x_news, y_news, benchmark, iteration, steps=1000):
    assert(isinstance(benchmark, SyntheticFunction))

    # data for visualization
    x_test = jnp.linspace(*benchmark.bounds, steps)
    y_test = benchmark.f(x_test)

    fig, ax = plt.subplots()
    fig.set_dpi(300)
    ax.scatter(x_train, y_train, marker='o', label="Observations", color=cols[0])
    ax.plot(x_test, y_test, label="Latent function", color=cols[0], linewidth=2)
    ax.scatter(x_news, y_news, marker='X', label="New points", color='r')
    ax.legend()
    plt.title(f"Iteration: {iteration+1}")
    fig_name = f"../results/MF-UCB/1D/0/Iter_{iteration+1}_results.svg"
    fig.savefig(fig_name)
    plt.close()

def plot_results_2D(X_train, X_news, benchmark, n_init, iteration, steps=1000):
    assert(isinstance(benchmark, SyntheticFunction))
    bounds = benchmark.bounds
    # data for visualization
    x1 = torch.linspace(bounds[0, 0].item(), bounds[1, 0].item(), steps)
    x2 = torch.linspace(bounds[0, 1].item(), bounds[1, 1].item(), steps)
    grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing='ij')
    X_test = torch.cat((grid_x1.reshape(-1, 1), grid_x2.reshape(-1, 1)), 1)
    y_test = benchmark.f(X_test)
        
    X_news = jnp.array(X_news)
    plt.figure(dpi=300)
    plt.contourf(grid_x1, grid_x2, -y_test.reshape(steps, steps), levels=20, cmap='viridis')
    plt.colorbar(label='Function Value')
    plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', label="Observations", color='orange')
    plt.scatter(X_news[:, 0], X_news[:, 1], marker='X', label="New points", color='r')
    
    # Annotate points with numbers
    for i, _ in enumerate(X_train):
        if i <  n_init:
            plt.annotate(f'0', (X_train[i, 0], X_train[i, 1]), textcoords="offset points", xytext=(0, -2.5), ha='center', fontsize=6)
        else:
            plt.annotate(f'{i-n_init+1}', (X_train[i, 0], X_train[i, 1]), textcoords="offset points", xytext=(0, -2.5), ha='center', fontsize=6)
    for j, _ in enumerate(X_news):    
        plt.annotate(f'new point', (X_news[j, 0], X_news[j, 1]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=5)
    
    plt.title(f"Iteration:{iteration+1}")
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    fig_name = f"../results/j-ATS-UCB/Branin/0/Iter_{iteration+1}_results.svg"
    plt.savefig(fig_name)
    plt.close()

def plot_mf_results_2D(X_train, X_news, benchmark, n_init, iteration, steps=1000):
    assert(isinstance(benchmark, SyntheticFunction))
    bounds = benchmark.bounds
    # data for visualization
    x1 = torch.linspace(bounds[0, 0].item(), bounds[1, 0].item(), steps)
    x2 = torch.linspace(bounds[0, 1].item(), bounds[1, 1].item(), steps)
    grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing='ij')
    X_test = torch.cat((grid_x1.reshape(-1, 1), grid_x2.reshape(-1, 1), torch.ones((steps*steps, 1))), 1)
    y_test = benchmark.f(X_test)
    
    fig, ax = plt.subplots()
    fig.set_dpi(300)
    func = ax.contourf(grid_x1, grid_x2, -y_test.reshape(steps, steps), levels=20, cmap='viridis')
    bar1 = plt.colorbar(func)
    fidelity_cmap = matplotlib.cm.get_cmap('Reds')
    ax.scatter(X_train[:n_init, 0], X_train[:n_init, 1], c=X_train[:n_init, -1], marker='s', label="initialization", cmap=fidelity_cmap)
    fidelity = ax.scatter(X_train[:, 0], X_train[:, 1], c=X_train[:, -1], marker='o', label="observation", cmap=fidelity_cmap)
    fidelity.set_clim(vmin=0.0, vmax=1.0)
    ax.scatter(X_news[:, 0], X_news[:, 1], c=fidelity_cmap(X_news[:, -1]), marker='X', label="next query")
    bar2 = plt.colorbar(fidelity, ticks=[k/10 for k in range(11)])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3)

    # Annotate points with numbers
    for i, _ in enumerate(X_train):
        if i < n_init:
            plt.annotate(f'0', (X_train[i, 0], X_train[i, 1]), textcoords="offset points", xytext=(0, -3), ha='center', fontsize=8)
        else:
            plt.annotate(f'{i-n_init+1}', (X_train[i, 0], X_train[i, 1]), textcoords="offset points", xytext=(0, -3), ha='center', fontsize=8)
    for j, _ in enumerate(X_news):    
        plt.annotate(f'new point', (X_news[j, 0], X_news[j, 1]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=5)    
    plt.title(f"Iteration:{iteration+1}", loc='center', pad=30)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.xticks([k-5 for k in range(16)])
    plt.yticks(list(range(16)))
    bar1.set_label("Function Value at the Highest Fidelity")
    bar2.set_label("Fidelity")
    plt.tight_layout()
    fig_name = f"../results/MF-UCB/cj-UCB-3/AugmentedBranin/1/Iter_{iteration+1}_results.svg"
    fig.savefig(fig_name)
    plt.close()

def plot_mf_final_2D(X_train, X_final, benchmark, n_init, steps=1000):
    assert(isinstance(benchmark, SyntheticFunction))
    bounds = benchmark.bounds
    # data for visualization
    x1 = torch.linspace(bounds[0, 0].item(), bounds[1, 0].item(), steps)
    x2 = torch.linspace(bounds[0, 1].item(), bounds[1, 1].item(), steps)
    grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing='ij')
    X_test = torch.cat((grid_x1.reshape(-1, 1), grid_x2.reshape(-1, 1), torch.ones((steps*steps, 1))), 1)
    y_test = benchmark.f(X_test)
    
    X_final = jnp.concatenate([X_final, jnp.array([1.])], axis=0).reshape(1, -1)
    fig, ax = plt.subplots()
    fig.set_dpi(300)
    func = ax.contourf(grid_x1, grid_x2, -y_test.reshape(steps, steps), levels=20, cmap='viridis')
    bar1 = plt.colorbar(func)
    fidelity_cmap = matplotlib.cm.get_cmap('Reds')
    ax.scatter(X_train[:n_init, 0], X_train[:n_init, 1], c=X_train[:n_init, -1], marker='s', label="initialization", cmap=fidelity_cmap)
    fidelity = ax.scatter(X_train[:, 0], X_train[:, 1], c=X_train[:, -1], marker='o', label="observation", cmap=fidelity_cmap)
    fidelity.set_clim(vmin=0.0, vmax=1.0)
    bar2 = plt.colorbar(fidelity)
    ax.scatter(X_final[:, 0], X_final[:, 1], c=fidelity_cmap(X_final[:, -1]), marker='X', label="final recommendation")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3)

    # Annotate points with numbers
    for i, _ in enumerate(X_train):
        if i < n_init:
            plt.annotate(f'0', (X_train[i, 0], X_train[i, 1]), textcoords="offset points", xytext=(0,-3), ha='center', fontsize=8)
        else:
            plt.annotate(f'{i-n_init+1}', (X_train[i, 0], X_train[i, 1]), textcoords="offset points", xytext=(0,-3), ha='center', fontsize=8)
    plt.annotate(f'final', (X_final[:, 0], X_final[:, 1]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=6)
    plt.title(r"baseline", loc='center', pad=30)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.xticks([k-5 for k in range(16)])
    plt.yticks(list(range(16)))
    bar1.set_label("Function Value at the Highest Fidelity")
    bar2.set_label("Fidelity")
    plt.tight_layout()
    fig_name = f"./results/final_results.svg"
    fig.savefig(fig_name)
    plt.close()

def plot_cost_evaluation(results, n_init):
    fidelity = results[n_init:, -3]
    y = jnp.array(-1.) * results[n_init:, -2]
    cost = results[:, -1]
    cumulative_cost = jnp.cumsum(cost)[n_init:]
    
    fig, ax = plt.subplots()
    fig.set_dpi(300)
    fidelity_cmap = matplotlib.cm.get_cmap('viridis')
    fidelity = ax.scatter(cumulative_cost, y, c=fidelity, marker='o', cmap=fidelity_cmap)
    fidelity.set_clim(vmin=0.0, vmax=1.0)
    bar = plt.colorbar(fidelity)
    
    plt.title(r"$\beta(z)=1$, $\gamma(z)=exp(z-z_{\bullet})$")
    plt.xlabel('Cumulative Cost')
    plt.ylabel('Evaluation')
    bar.set_label("Fidelity")
    fig_name = f"../results/MF-UCB/cj-UCB-4/1D/0/cost_evaluation.svg"
    fig.savefig(fig_name)
import jax.numpy as jnp
import torch
import json

from script.model import GPModel
from script.GP_optimizer import BayOptimizer
from test_functions import simple_1d, func_wrapper, SyntheticFunction

from matplotlib import pyplot as plt
import matplotlib

cols = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]

def plot_GPs(model, x_train, y_train, benchmark, iteration, batch, batch_size=1, steps=1000):
    """
    plot a series of Gaussian process
    """
    assert(isinstance(model, BayOptimizer)), "`model` must be an instance of BayOptimizer"
    assert(isinstance(benchmark, SyntheticFunction))

    # data for visualization
    x_test = jnp.linspace(*benchmark.bounds, steps)
    y_test = benchmark.f(x_test)

    x_pred_mean = jnp.zeros((steps, ))
    x_pred_std = jnp.zeros(steps, )

    ## Prediction (just for visualization purpose)
    for i, sm in enumerate(model.SMs):
        pred_mean, pred_std = sm.inference(x_test)
        
        
        x_pred_mean += pred_mean
        x_pred_std += pred_std
        #acqs_test = pred_mean + model.j * pred_std
        #acqs_test = jnp.array([model.acqs_avg(x, model.j) for x in x_test])  # slow

        # visualization
        fig, ax = plt.subplots()
        fig.set_dpi(300)
        ax.scatter(x_train, y_train, marker='o', label="Observations", color=cols[0], alpha=0.5)
        ax.plot(x_test, y_test, label="Latent function", color=cols[0], linewidth=2)
        ax.plot(x_test, pred_mean, label="Predictive mean", color=cols[1])
        ax.fill_between(x_test.squeeze(), pred_mean-2*pred_std, pred_mean+2*pred_std, label="$2\sigma$", alpha=0.5, color=cols[1])
        #ax.scatter(x_new, y_new, marker='X', label="New point", color='r')
        #ax.plot(x_test, acqs_test, label="Acquisition function", color=cols[2])
        ax.legend()
        #plt.show()
        if batch_size > 1:
            plt.title(f"Iteration: {iteration+1} Batch: {batch+1} GP: {i+1}")
            fig_name = f"../results/j-ATS-UCB/1D/0/GP/Iter_{iteration+1}_Batch_{batch+1}_GP_{i+1}.svg"
        else:
            plt.title(f"Iteration: {iteration+1} GP: {i+1}")
            fig_name = f"../results/j-ATS-UCB/1D/0/GP/Iter_{iteration+1}_GP_{i+1}.svg"
        fig.savefig(fig_name)
        plt.close()

    acqs_test = (x_pred_mean + model.j * x_pred_std)/len(model.SMs)

    file_path = f"../data/j-ATS-UCB/1D/0/acqs_func/simple_4_iteration_{iteration}_batch_{batch}.json"
    with open(file_path, 'w') as f:
        json.dump(acqs_test.tolist(), f)
    return acqs_test

def plot_acqs_funcs(model, acqs_tests, x_train, y_train, x_news, y_news, benchmark, iteration, batch_size, steps=1000):
    assert(isinstance(model, BayOptimizer)), "`model` must be an instance of BayOptimizer"
    assert(isinstance(benchmark, SyntheticFunction))
    assert(batch_size > 1)

    # data for visualization
    x_test = jnp.linspace(*benchmark.bounds, steps)
    y_test = benchmark.f(x_test)

    acqs_tests = jnp.array(acqs_tests)

    fig, ax = plt.subplots()
    fig.set_dpi(300)
    ax.scatter(x_train, y_train, marker='o', label="Observations", color=cols[0], alpha=0.5)
    ax.plot(x_test, y_test, label="Latent function", color=cols[0], linewidth=2)
    for i in range(batch_size):
        ax.scatter(x_news[i], y_news[i], marker='X', label=f"new point {i+1}", color=cols[i+1])
        ax.plot(x_test, acqs_tests[i, :], label=f"acquisition function {i+1}", color=cols[i+1], linewidth=2)
    ax.legend()
    #plt.title(f"Acquisition functions of iteration: {iteration+1}")
    fig_name = f"../results/j-ATS-UCB/1D/0/acqs_func/Iter_{iteration+1}.svg"
    fig.savefig(fig_name)
    plt.close()

def plot_results_1D(x_train, y_train, x_news, y_news, benchmark, iteration, steps=1000):
    assert(isinstance(benchmark, SyntheticFunction))

    # data for visualization
    x_test = jnp.linspace(*benchmark.bounds, steps)
    y_test = benchmark.f(x_test)

    fig, ax = plt.subplots()
    fig.set_dpi(300)
    ax.scatter(x_train, y_train, marker='o', label="Observations", color=cols[0])
    ax.plot(x_test, y_test, label="Latent function", color=cols[0], linewidth=2)
    ax.scatter(x_news, y_news, marker='X', label="New points", color='r')
    ax.legend()
    plt.title(f"Iteration: {iteration+1}")
    fig_name = f"../results/MF-UCB/1D/0/Iter_{iteration+1}_results.svg"
    fig.savefig(fig_name)
    plt.close()

def plot_results_2D(X_train, X_news, benchmark, n_init, iteration, steps=1000):
    assert(isinstance(benchmark, SyntheticFunction))
    bounds = benchmark.bounds
    # data for visualization
    x1 = torch.linspace(bounds[0, 0].item(), bounds[1, 0].item(), steps)
    x2 = torch.linspace(bounds[0, 1].item(), bounds[1, 1].item(), steps)
    grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing='ij')
    X_test = torch.cat((grid_x1.reshape(-1, 1), grid_x2.reshape(-1, 1)), 1)
    y_test = benchmark.f(X_test)
        
    X_news = jnp.array(X_news)
    plt.figure(dpi=300)
    plt.contourf(grid_x1, grid_x2, -y_test.reshape(steps, steps), levels=20, cmap='viridis')
    plt.colorbar(label='Function Value')
    plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', label="Observations", color='orange')
    plt.scatter(X_news[:, 0], X_news[:, 1], marker='X', label="New points", color='r')
    
    # Annotate points with numbers
    for i, _ in enumerate(X_train):
        if i <  n_init:
            plt.annotate(f'0', (X_train[i, 0], X_train[i, 1]), textcoords="offset points", xytext=(0, -2.5), ha='center', fontsize=6)
        else:
            plt.annotate(f'{i-n_init+1}', (X_train[i, 0], X_train[i, 1]), textcoords="offset points", xytext=(0, -2.5), ha='center', fontsize=6)
    for j, _ in enumerate(X_news):    
        plt.annotate(f'new point', (X_news[j, 0], X_news[j, 1]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=5)
    
    plt.title(f"Iteration:{iteration+1}")
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    fig_name = f"../results/j-ATS-UCB/Branin/0/Iter_{iteration+1}_results.svg"
    plt.savefig(fig_name)
    plt.close()

def plot_mf_results_2D(X_train, X_news, benchmark, n_init, iteration, steps=1000):
    assert(isinstance(benchmark, SyntheticFunction))
    bounds = benchmark.bounds
    # data for visualization
    x1 = torch.linspace(bounds[0, 0].item(), bounds[1, 0].item(), steps)
    x2 = torch.linspace(bounds[0, 1].item(), bounds[1, 1].item(), steps)
    grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing='ij')
    X_test = torch.cat((grid_x1.reshape(-1, 1), grid_x2.reshape(-1, 1), torch.ones((steps*steps, 1))), 1)
    y_test = benchmark.f(X_test)
    
    fig, ax = plt.subplots()
    fig.set_dpi(300)
    func = ax.contourf(grid_x1, grid_x2, -y_test.reshape(steps, steps), levels=20, cmap='viridis')
    bar1 = plt.colorbar(func)
    fidelity_cmap = matplotlib.cm.get_cmap('Reds')
    ax.scatter(X_train[:n_init, 0], X_train[:n_init, 1], c=X_train[:n_init, -1], marker='s', label="initialization", cmap=fidelity_cmap)
    fidelity = ax.scatter(X_train[:, 0], X_train[:, 1], c=X_train[:, -1], marker='o', label="observation", cmap=fidelity_cmap)
    fidelity.set_clim(vmin=0.0, vmax=1.0)
    ax.scatter(X_news[:, 0], X_news[:, 1], c=fidelity_cmap(X_news[:, -1]), marker='X', label="next query")
    bar2 = plt.colorbar(fidelity, ticks=[k/10 for k in range(11)])
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3)

    # Annotate points with numbers
    for i, _ in enumerate(X_train):
        if i < n_init:
            plt.annotate(f'0', (X_train[i, 0], X_train[i, 1]), textcoords="offset points", xytext=(0, -3), ha='center', fontsize=8)
        else:
            plt.annotate(f'{i-n_init+1}', (X_train[i, 0], X_train[i, 1]), textcoords="offset points", xytext=(0, -3), ha='center', fontsize=8)
    for j, _ in enumerate(X_news):    
        plt.annotate(f'new point', (X_news[j, 0], X_news[j, 1]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=5)    
    plt.title(f"Iteration:{iteration+1}", loc='center', pad=30)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.xticks([k-5 for k in range(16)])
    plt.yticks(list(range(16)))
    bar1.set_label("Function Value at the Highest Fidelity")
    bar2.set_label("Fidelity")
    plt.tight_layout()
    fig_name = f"../results/MF-UCB/cj-UCB-3/AugmentedBranin/1/Iter_{iteration+1}_results.svg"
    fig.savefig(fig_name)
    plt.close()

def plot_mf_final_2D(X_train, X_final, benchmark, n_init, steps=1000):
    assert(isinstance(benchmark, SyntheticFunction))
    bounds = benchmark.bounds
    # data for visualization
    x1 = torch.linspace(bounds[0, 0].item(), bounds[1, 0].item(), steps)
    x2 = torch.linspace(bounds[0, 1].item(), bounds[1, 1].item(), steps)
    grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing='ij')
    X_test = torch.cat((grid_x1.reshape(-1, 1), grid_x2.reshape(-1, 1), torch.ones((steps*steps, 1))), 1)
    y_test = benchmark.f(X_test)
    
    X_final = jnp.concatenate([X_final, jnp.array([1.])], axis=0).reshape(1, -1)
    fig, ax = plt.subplots()
    fig.set_dpi(300)
    func = ax.contourf(grid_x1, grid_x2, -y_test.reshape(steps, steps), levels=20, cmap='viridis')
    bar1 = plt.colorbar(func)
    fidelity_cmap = matplotlib.cm.get_cmap('Reds')
    ax.scatter(X_train[:n_init, 0], X_train[:n_init, 1], c=X_train[:n_init, -1], marker='s', label="initialization", cmap=fidelity_cmap)
    fidelity = ax.scatter(X_train[:, 0], X_train[:, 1], c=X_train[:, -1], marker='o', label="observation", cmap=fidelity_cmap)
    fidelity.set_clim(vmin=0.0, vmax=1.0)
    bar2 = plt.colorbar(fidelity)
    ax.scatter(X_final[:, 0], X_final[:, 1], c=fidelity_cmap(X_final[:, -1]), marker='X', label="final recommendation")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3)

    # Annotate points with numbers
    for i, _ in enumerate(X_train):
        if i < n_init:
            plt.annotate(f'0', (X_train[i, 0], X_train[i, 1]), textcoords="offset points", xytext=(0,-3), ha='center', fontsize=8)
        else:
            plt.annotate(f'{i-n_init+1}', (X_train[i, 0], X_train[i, 1]), textcoords="offset points", xytext=(0,-3), ha='center', fontsize=8)
    plt.annotate(f'final', (X_final[:, 0], X_final[:, 1]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=6)
    plt.title(r"baseline", loc='center', pad=30)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.xticks([k-5 for k in range(16)])
    plt.yticks(list(range(16)))
    bar1.set_label("Function Value at the Highest Fidelity")
    bar2.set_label("Fidelity")
    plt.tight_layout()
    fig_name = f"./results/final_results.svg"
    fig.savefig(fig_name)
    plt.close()

def plot_cost_evaluation(results, n_init):
    fidelity = results[n_init:, -3]
    y = jnp.array(-1.) * results[n_init:, -2]
    cost = results[:, -1]
    cumulative_cost = jnp.cumsum(cost)[n_init:]
    
    fig, ax = plt.subplots()
    fig.set_dpi(300)
    fidelity_cmap = matplotlib.cm.get_cmap('viridis')
    fidelity = ax.scatter(cumulative_cost, y, c=fidelity, marker='o', cmap=fidelity_cmap)
    fidelity.set_clim(vmin=0.0, vmax=1.0)
    bar = plt.colorbar(fidelity)
    
    plt.title(r"$\beta(z)=1$, $\gamma(z)=exp(z-z_{\bullet})$")
    plt.xlabel('Cumulative Cost')
    plt.ylabel('Evaluation')
    bar.set_label("Fidelity")
    fig_name = f"../results/MF-UCB/cj-UCB-4/1D/0/cost_evaluation.svg"
    fig.savefig(fig_name)
    plt.close()