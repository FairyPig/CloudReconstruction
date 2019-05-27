import os
import random

render_para_path = './render_para.txt'
render_para = open(render_para_path, 'w')

azimuth_pace = 30
elevation_pace = 30
sun_intensity_low = 300
cloud_density_low = 4.0

azimuth_num = 6
elevation_num = 2
sun_intensity_num = 3
cloud_density_num = 4

for azimuth in range(azimuth_num):
    for elevation in range(elevation_num):
        for sun_intensity in range(sun_intensity_num):
            for cloud_density in range(cloud_density_num):
                if azimuth != 2 and azimuth != 3:
                    print(azimuth)
                    a = str(15 + azimuth_pace * azimuth)
                    e = str(elevation_pace * (elevation + 1))
                    sun_inten = str(sun_intensity_low + sun_intensity * 100)
                    cloud_inten = str(cloud_density_low + cloud_density * 0.5)
                    m_str = a + ' ' + e + ' ' + sun_inten + ' ' + cloud_inten + '\n'
                    render_para.write(m_str)
render_para.close()
