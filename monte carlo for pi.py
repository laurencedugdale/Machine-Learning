# This program uses a Monte Carlo simulation to calculate an approximate value for pi. It works by defining a circle contained within a square, where the outside
# of the circle makes contact with the edges of the square. Points are randomly generated within the square, and a record is made of the number of points that
# fall within the circle compared to those that fall outside. This provides the approximate relative areas of the two shapes. The area of the square is known,
# so the area of the cirlce can be estimated. This area can then be used to provide an approximate vaule for pi by rearranging the equation for the area of a circle.



import random
outside=0
inside=0
for iteration in range(100000):
    length=(((random.uniform(-0.5,0.5))**2)+((random.uniform(-0.5,0.5))**2))**0.5
    if length>0.5:
        outside=outside+1
    else:
        inside=inside+1
pi_approx=(inside/(outside+inside))/(0.5**2)
print('True value of pi is 3.141592...')
print('Calculated approximate vaule of pi is ',pi_approx)
