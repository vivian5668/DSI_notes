from math import pi

def circle_area(r):
	if type(r) not in [int, float]:
		raise TypeError("Please only input an integer")
	if r < 0:
		raise ValueError("the radius can not be negative")
	return pi * r ** 2
