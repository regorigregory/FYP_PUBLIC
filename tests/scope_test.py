city = "Kansas"
print("Initial value of the city: {0}".format(city))
def f():
    city = "London"
    print("City after calling f: {0}".format(city))

    def g():
        nonlocal city
        city = "Tijuana"
        print("City after calling g: {0}".format(city))
        def h():
            nonlocal city
            city = "New Mexico"
            print("City after calling h: {0}".format(city))
        h()
    g()
    print("City after calling g from f: {0}".format(city))


f()
print("City after calling everything in the global scope: {0}".format(city))


