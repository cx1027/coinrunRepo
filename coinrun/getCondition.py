import numpy as np
from coinrun.coinrun import setup_utils, make
import imageSave as img
import matplotlib.pyplot as plt
import random


def getCondition(num_envs=1, max_steps=100000):
    #random environment
    # setup_utils.setup_and_load(use_cmd_line_args=False)
    #just test in level1 with config --run-id myrun --num-levels 1
    setup_utils.setup_and_load()
    env = make('standard', num_envs=num_envs)
    imgNum = 0
    for step in range(100000):
        env.render()
        #acts = np.array([env.action_space.sample() for _ in range(env.num_envs)])

        foo=[1,3]
        acts = np.array([random.choice(foo)])
        #0: no move
        #1:right move
        #2: move but stay
        #3:jump
        #4:down
        #5:down
        #6:down

        # 0, 0,
        # +1, 0, // right
        # -1, 0, // left
        # 0, +1, // jump
        # +1, +1, // right - jump
        # -1, +1, // left - jump
        # 0, -1, // down(step down from a crate)

        print("python input action: ", acts)
        print("\n env.step(acts): \n")
        _obs, rews, _dones, _infos = env.step(acts)
        #todo:return distance (change _obs to distance) then condition

        # arr = env.getArray()
        aaa = env.getCondition()
        env.getConditionArray()

        print("aaa bbb:")
        print(np.array(aaa))



        img_input = img.imgbuffer_process(_obs, (256, 256))

        if step % 50 == 0:
            #turn gray
            #todo:make coinrunMOXCS consume gray img
            #plt.imsave('%i.jpg' % (imgNum), img_input.mean(axis=2), cmap = "gray")
            # plt.imsave('%i.jpg' % (imgNum), img_input)
            #plt.imshow(img_input.mean(axis=2), cmap="gray")
            imgNum = imgNum + 1
            print("imgNum:%i" % (imgNum))


        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
        print("step", step, "rews", rews)

    env.close()


if __name__ == '__main__':
    random_agent()