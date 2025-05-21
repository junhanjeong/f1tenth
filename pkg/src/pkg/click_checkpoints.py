# python click_checkpoints.py로 실행
# 이미지를 클릭하면 해당 좌표를 출력

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('maps/map_easy3.png')  # 트랙 이미지 경로
fig, ax = plt.subplots()
ax.imshow(img)

coords = []

def onclick(event):
    ix, iy = int(event.xdata), int(event.ydata)
    print('x = %d, y = %d' % (ix, iy))
    coords.append([ix, iy])

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# 클릭한 좌표는 coords에 순서대로 저장됨
