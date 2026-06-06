r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


CUSTOM_DATA_URL = "https://s3.amazonaws.com/fast-ai-imageclas/stanford-cars.tgz"


def vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 32
    hypers['h_dim'] = 256
    hypers['z_dim'] = 128
    hypers['x_sigma2'] = 0.5
    hypers['learn_rate'] = 1e-3
    hypers['betas'] = (0.9, 0.999)
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**

"""

part2_q2 = r"""
**Your answer:**


"""

part2_q3 = r"""
**Your answer:**

"""

part2_q4 = r"""
**Your answer:**

"""

# ==============
