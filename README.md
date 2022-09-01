run main.py

change (-100, 100, (100,3)) to adjust the number of Vertex.
'''
pts = np.random.randint(-100, 100, (100,3))
'''

The frames could be produced with `ffmpeg`:
```
$ffmpeg -framerate 8 -pattern_type glob -i './frames/*.png' -c:v libx264 -r 30 -pix_fmt yuv420p example.mp4
```

