import mnist, numpy, time
import matplotlib.pyplot as plt
import seaborn as sns
from contrastive_divergence import *

sns.set_style("white")

if __name__ == "__main__":
    # set a random seed so that results are reproducible
    numpy.random.seed(6786976)
    
    # load in the images
    labels, images, dim = mnist.load("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz")

    # run contrastive divergence
    minibatch_size = 200
    n_hidden = 1000
    n_contrastive_divergence_steps = 3
    momentum = 0.0
    n_epochs = 10
    method = "RMSprop"

    start = time.time()
    a,b,W,mem = descent(images, minibatch_size, dim, n_hidden, n_contrastive_divergence_steps, momentum, n_epochs, method = method)
    end = time.time()
    print("Training took {0:.2f} seconds".format(end - start))
    
    # plot the reconstruction error
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    sns.pointplot(x = numpy.arange(len(mem)), y = mem, color = 'b', ax = ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.xlabel('Training step', fontsize=18)
    plt.ylabel('Reconstruction error', fontsize=18)
    
    plt.tight_layout()
    fig.savefig('reconstruction_error_RMSprop.png')
    plt.close(fig)
        
    # plot a few reconstructions starting from data
    fig, axes = plt.subplots(nrows=3, ncols = 3)

    for i in range(3):
        randv = mnist.stochastic_binarize(mnist.minibatch(images, 1), 255)
        h_data, v_free, h_free = gibbs(randv, a, b, W, 1) # note that this is just one encoding-decoding step
        
        sns.heatmap(mnist.squareform(randv, dim), ax = axes[i][0], cbar = False, xticklabels = False, yticklabels = False)    
        sns.heatmap(mnist.squareform(v_free, dim), ax = axes[i][1], cbar = False, xticklabels = False, yticklabels = False)
        sns.heatmap(mnist.squareform(logistic(visible_fields(h_data, W, a)), dim), ax = axes[i][2], cbar = False, xticklabels = False, yticklabels = False)    
        
        axes[i][0].set_title('input', fontsize = 12)
        axes[i][1].set_title('stochastic reconstruction', fontsize = 12)
        axes[i][2].set_title('visible probabilities', fontsize = 12)

    plt.tight_layout()
    fig.savefig('reconstructions_RMSprop.png')
    plt.close(fig)
    plt.clf()
    
    # plot a few samples starting from random points
    fig, axes = plt.subplots(nrows=3, ncols=3)

    for i in range(3):
        randv =  bernoulli.rvs(0.2,size = dim[0]*dim[1])
        h_data, v_free, h_free = gibbs(randv, a, b, W, 100) # run 100 Monte Carlo steps in hopes of reaching an equilibrium
        
        sns.heatmap(mnist.squareform(randv, dim), ax = axes[i][0], cbar = False, xticklabels = False, yticklabels = False)    
        sns.heatmap(mnist.squareform(v_free, dim), ax = axes[i][1], cbar = False, xticklabels = False, yticklabels = False)
        sns.heatmap(mnist.squareform(logistic(visible_fields(h_data, W, a)), dim), ax = axes[i][2], cbar = False, xticklabels = False, yticklabels = False)    
        
        axes[i][0].set_title('input', fontsize = 12)
        axes[i][1].set_title('stochastic sample', fontsize = 12)
        axes[i][2].set_title('visible probabilities', fontsize = 12)
        
    plt.tight_layout()
    fig.savefig('samples_RMSprop.png')
    plt.close(fig)
    plt.clf()
    
    # plot a few of the local energy minima starting from random points 
    fig, axes = plt.subplots(nrows=3, ncols=3)

    for i in range(3):
        randv =  bernoulli.rvs(0.2,size = dim[0]*dim[1])
        h_data, v_free, h_free = anneal(randv, a, b, W, 100, lambda x: 10.0*(0.9**x)) # run 100 Monte Carlo steps in hopes of reaching an equilibrium
        
        sns.heatmap(mnist.squareform(randv, dim), ax = axes[i][0], cbar = False, xticklabels = False, yticklabels = False)    
        sns.heatmap(mnist.squareform(v_free, dim), ax = axes[i][1], cbar = False, xticklabels = False, yticklabels = False)
        sns.heatmap(mnist.squareform(logistic(visible_fields(h_data, W, a)), dim), ax = axes[i][2], cbar = False, xticklabels = False, yticklabels = False)    
        
        axes[i][0].set_title('input', fontsize = 12)
        axes[i][1].set_title('stochastic sample', fontsize = 12)
        axes[i][2].set_title('visible probabilities', fontsize = 12)
        
    plt.tight_layout()
    fig.savefig('annealed_samples_RMSprop.png')
    plt.close(fig)
    plt.clf()
