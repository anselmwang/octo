
import itertools
import datasets
import tensorflow_datasets as tfds
hf_ds = datasets.load_dataset("jxu124/OpenX-Embodiment", "austin_buds_dataset_converted_externally_to_rlds", streaming=True, split='train')  # IterDataset
print(type(hf_ds)) # <class 'datasets.iterable_dataset.IterableDataset'>

berkeley_tf_ds = tfds.load("berkeley_mvp_converted_externally_to_rlds", data_dir="gs://gresearch/robotics", split="train")
berkeley_traj_tf = next(iter(berkeley_tf_ds))
berkeley_steps = list(berkeley_traj_tf['steps'])


rt1_tf_ds = tfds.load("fractal20220817_data", data_dir="gs://gresearch/robotics", split="train")
for episode_no, rt1_traj_tf in enumerate(itertools.islice(rt1_tf_ds, 100)):
    rt1_steps = list(rt1_traj_tf['steps'])
    tps= [list(rt1_step["action"]["terminate_episode"].numpy()) for rt1_step in rt1_steps]
    for step_no, rt1_step in enumerate(rt1_steps):
        reward = float(rt1_step["reward"]) 
        if reward != 0:
            print(f"ep{episode_no} reward{reward} at {step_no}/{len(rt1_steps)}")



# test_df = tfds.load('cifar10', split="train")
tf_ds = tfds.load("austin_buds_dataset_converted_externally_to_rlds", data_dir="gs://gresearch/robotics", split="train")
print(type(tf_ds)) # <class 'tensorflow.python.data.ops.prefetch_op._PrefetchDataset'>

for traj_hf, traj_tf in itertools.islice(zip(hf_ds, tf_ds), 10):
    print(traj_hf["data.pickle"].keys(), traj_tf.keys())
    print(len(traj_hf['data.pickle']['steps']), len(list(traj_tf['steps'])))
    print(traj_hf['data.pickle']['steps'][0].keys(), list(traj_tf['steps'])[0].keys())

a = 3
# %% [markdown]
# # Octo Dataloading Examples
# 
# This notebook will walk you through some of the primary features of the Octo dataloader. Data is, after all, the most important part of any machine learning pipeline!

# %% [markdown]
# ## Loading Open X-Embodiment Data
# 
# The [Open X-Embodiment (OXE)](https://robotics-transformer-x.github.io/) project was a massive cross-instutition data collection collaboration the likes of which robot learning has never seen before. The resulting dataset includes 22 different robots demonstrating 527 skills and totals over 1 million trajectories. However, as we found throughout the course of the Octo project, simply loading such a diverse set of robot data is no small feat. We hope that the `octo.data` pipeline can help kickstart anyone who hopes to take advantage of the massive collection of robot demonstrations that is OXE!
# 
# ### Minimum working example to load a single OXE dataset

# %%
import tensorflow_datasets as tfds

# tfds.load("austin_buds_dataset_converted_externally_to_rlds", data_dir="gs://gresearch/robotics")

ds = tfds.load("droid_100", data_dir="gs://gresearch/robotics", split="train")

# episodes = list(ds.shuffle(10, seed=0).take(10))
episodes = list(ds)

episodes_in_steps = [list(episode["steps"]) for episode in episodes]

# %%

def verify(steps):
    discounts = []
    is_firsts = []
    is_lasts = []
    is_terminals = []
    for x in steps:
        discounts.append(int(x["discount"].numpy()))
        is_firsts.append(int(x["is_first"].numpy()))
        is_lasts.append(int(x["is_last"].numpy()))
        is_terminals.append(int(x["is_terminal"].numpy()))

    assert all(discount == 1 for discount in discounts) 
    assert is_firsts[0] == 1 and all(is_first == 0 for is_first in is_firsts[1:])
    assert is_lasts[-1] == 1 and all(is_last == 0 for is_last in is_lasts[:-1])
    assert is_terminals[-1] == 1 and all(is_terminal == 0 for is_terminal in is_terminals[:-1])

for steps in episodes_in_steps:
    verify(steps)

# %%

# %%
# minimum working example to load a single OXE dataset
from octo.data.oxe import make_oxe_dataset_kwargs
from octo.data.dataset import make_single_dataset

dataset_kwargs = make_oxe_dataset_kwargs(
    # see octo/data/oxe/oxe_dataset_configs.py for available datasets
    # (this is a very small one for faster loading)
    "austin_buds_dataset_converted_externally_to_rlds",
    # can be local or on cloud storage (anything supported by TFDS)
    # "/path/to/base/oxe/directory",
    "gs://gresearch/robotics",
    load_camera_views=("primary", "wrist"),
    # load_depth=True,
    load_proprio=True

)
dataset = make_single_dataset(dataset_kwargs, train=True) # load the train split
iterator = dataset.iterator()

# %%
# make_single_dataset yields entire trajectories
traj = next(iterator)
print("Top-level keys: ", traj.keys())
print("Observation keys: ", traj["observation"].keys())
traj["observation"]["image_wrist"].shape
traj["observation"]["proprio"].shape
traj["observation"]["proprio"][0]
traj["observation"]["timestep"].shape
traj["observation"]["timestep"][:5]
traj["observation"]["pad_mask_dict"].keys()
traj["observation"]["task_completed"].shape
print("Task keys: ", traj["task"].keys())
traj["task"]["language_instruction"].shape
traj["task"]["pad_mask_dict"].keys()

# %%
from PIL import Image
import numpy as np

traj = next(iterator)
images = traj["observation"]["image_primary"]
# should be: (traj_len, window_size, height, width, channels)
# (window_size defaults to 1)
print(images.shape)  
Image.fromarray(np.concatenate(images.squeeze()[-5:], axis=1))

# %%
# you should set these much higher in practice (as large as your memory can hold!)
SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 64

# turning a dataset of trajectories into a training-ready batched dataset
train_dataset = (
    dataset.flatten() # flattens trajectories into individual frames
    .shuffle(SHUFFLE_BUFFER_SIZE) # shuffles the frames
    .batch(BATCH_SIZE) # batches the frames
)
batch = next(train_dataset.iterator())
images = batch["observation"]["image_primary"]
# should be: (batch_size, window_size, height, width, channels)
print(images.shape)
Image.fromarray(np.concatenate(images.squeeze()[:5], axis=1))

# %% [markdown]
# ### Loading a training-ready OXE mix
# 
# In reality, you're probably going to want to mix multiple datasets together, as well as use other transformations such as resizing, augmentation, windowing, etc. This section will show you how to get a proper OXE mix up and running, as well as demonstrate additional `octo.data` features for more realistic use-cases.

# %%
from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.data.dataset import make_interleaved_dataset

dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
    # you can pass your own list of dataset names and sample weights here, but we've
    # also provided a few named mixes for convenience. The Octo model was trained
    # using the "oxe_magic_soup" mix.
    "rtx",
    # can be local or on cloud storage (anything supported by TFDS)
    "gs://gresearch/robotics",
    # let's get a wrist camera!
    load_camera_views=("primary", "wrist"),
)

# see `octo.data.dataset.make_dataset_from_rlds` for the meaning of these kwargs
dataset_kwargs_list[0]

# %%
SHUFFLE_BUFFER_SIZE = 1000
BATCH_SIZE = 8

# each element of `dataset_kwargs_list` can be used with `make_single_dataset`, but let's
# use the more powerful `make_interleaved_dataset` to combine them for us!
dataset = make_interleaved_dataset(
    dataset_kwargs_list,
    sample_weights,
    train=True,
    # unlike our manual shuffling above, `make_interleaved_dataset` will shuffle
    # the JPEG-encoded images, so you should be able to fit a much larger buffer size
    shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    # see `octo.data.dataset.apply_trajectory_transforms` for full documentation
    # of these configuration options
    traj_transform_kwargs=dict(
        goal_relabeling_strategy="uniform",  # let's get some goal images
        window_size=2,  # let's get some history
        action_horizon=4,  # let's get some future actions for action chunking
        subsample_length=100,  # subsampling long trajectories improves shuffling a lot
    ),
    # see `octo.data.dataset.apply_frame_transforms` for full documentation
    # of these configuration options
    frame_transform_kwargs=dict(
        # let's apply some basic image augmentations -- see `dlimp.transforms.augment_image`
        # for full documentation of these configuration options
        image_augment_kwargs=dict(
            primary=dict(
                augment_order=["random_resized_crop", "random_brightness"],
                random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                random_brightness=[0.1],
            )
        ),
        # providing a `resize_size` is highly recommended for a mixed dataset, otherwise
        # datasets with different resolutions will cause errors
        resize_size=dict(
            primary=(256, 256),
            wrist=(128, 128),
        ),
        # If parallelism options are not provided, they will default to tf.Data.AUTOTUNE.
        # However, we would highly recommend setting them manually if you run into issues
        # with memory or dataloading speed. Frame transforms are usually the speed
        # bottleneck (due to image decoding, augmentation, and resizing), so you can set
        # this to a very high value if you have a lot of CPU cores. Keep in mind that more
        # parallel calls also use more memory, though.
        num_parallel_calls=64,
    ),
    # Same spiel as above about performance, although trajectory transforms and data reading
    # are usually not the speed bottleneck. One reason to manually set these is if you want
    # to reduce memory usage (since autotune may spawn way more threads than necessary).
    traj_transform_threads=16,
    traj_read_threads=16,
)

# Another performance knob to tune is the number of batches to prefetch -- again,
# the default of tf.data.AUTOTUNE can sometimes use more memory than necessary.
iterator = dataset.iterator(prefetch=1)

# %%
# phew, that was a lot of configuration! Let's see what we got.
batch = next(iterator)
print("Top-level keys: ", batch.keys())
# should now have "image_primary" and "image_wrist"!
print("Observation keys: ", batch["observation"].keys())
# should also have "image_primary" and "image_wrist", corresponding to future goal images
print("Task keys: ", batch["task"].keys())

# %%
from PIL import Image
import numpy as np

images_primary = batch["observation"]["image_primary"]
images_wrist = batch["observation"]["image_wrist"]
# should be: (batch_size, window_size (now 2), height, width, channels)
print(images_primary.shape)
print(images_wrist.shape)
actions = batch["action"]
# should be: (batch_size, window_size, action_horizon, action_dim)
print(actions.shape)

# %%
# let's visualize a window of primary images
display(Image.fromarray(np.concatenate(images_primary[0], axis=1)))
# now a window of wrist images -- many datasets don't have wrist images,
# so this will often be black
display(Image.fromarray(np.concatenate(images_wrist[0], axis=1)))
# pad_mask_dict also tells you which keys should be treated as padding
# (e.g., if the wrist camera is black, the corresponding pad_mask_dict entry is False)
print(batch["observation"]["pad_mask_dict"]["image_wrist"][0])

# %%
# let's take a look at the "task" dict: it should now have both goal
# images and language instructions!
goal_primary = batch["task"]["image_primary"]
goal_wrist = batch["task"]["image_wrist"]
language_instruction = batch["task"]["language_instruction"]
display(Image.fromarray(goal_primary[0]))
display(Image.fromarray(goal_wrist[0]))
print(language_instruction[0])


