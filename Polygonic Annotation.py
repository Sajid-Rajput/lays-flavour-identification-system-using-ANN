bbox = [1237.17528373266,2660.4035308953326,954.8297604035299,281.8327028163094]

sum_bbox = 0

segmentation = [1237.17528373266,2674.9810844892795,1387.8100042034457,2672.5514922236216,1399.957965531735,2718.713745271121,1827.5662042875144,2721.1433375367783,1825.1366120218565,2662.8331231609905,1958.7641866330378,2660.4035308953326,1956.3345943673799,2721.1433375367783,2192.00504413619,2730.86170659941,2189.575451870532,2886.3556116015116,1606.4733081126512,2881.4964270701958,1601.6141235813354,2942.236233711642,1557.8814627994946,2937.377049180326,1557.8814627994946,2883.9260193358537,1239.6048759983178,2866.918873476249]

segmentation_sum = 0

all_points_x = [1237.17528373266,1387.8100042034457,1399.957965531735,1827.5662042875144,1825.1366120218565,1958.7641866330378,1956.3345943673799,2192.00504413619,2189.575451870532,1606.4733081126512,1601.6141235813354,1557.8814627994946,1557.8814627994946,1239.6048759983178,1237.17528373266]

x_sum = 0

all_points_y = [2674.9810844892795,2672.5514922236216,2718.713745271121,2721.1433375367783,2662.8331231609905,2660.4035308953326,2721.1433375367783,2730.86170659941,2886.3556116015116,2881.4964270701958,2942.236233711642,2937.377049180326,2883.9260193358537,2866.918873476249,2674.9810844892795]

y_sum = 0


for i in range(0, len(bbox)):
    sum_bbox += bbox[i]

for i in range(0, len(segmentation)):
    segmentation_sum += segmentation[i]

for i in range(0, len(all_points_x)):
    x_sum += all_points_x[i]

for i in range(0, len(all_points_y)):
    y_sum += all_points_y[i]



print(f'bbox = {sum_bbox}')
print(f'segmentation = {segmentation_sum}')
print(f'all_points_x = {x_sum}')
print(f'all_points_y = {y_sum}')