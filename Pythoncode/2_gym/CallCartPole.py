import gymnasium as gym
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

saveAsPicture = False

def display_frames_as_gif(frames):
    plt.figure(figsize=(fames[0].shape[1]/72.0, frames[0].shape[0]/72.0),dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),interval=50)

    anim.save('movie_cartpole.mp4')
    display(display_animation(anim, default_mode='loop'))


if saveAsPicture == True:
    frames = []
    for _ in range(200):
        frames.append(env.render(mode='rgb_array'))
        action = np.random.choice(2)

        observation, reward, done, info = env.step(action)

    env.close()
else:
    for _ in range(1000):
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

