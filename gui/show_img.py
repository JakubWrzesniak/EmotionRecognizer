import pyglet


def show_img(title, path):
    try:
        image = pyglet.image.load(path)
        window = pyglet.window.Window(image.width, image.height, title)
        img = pyglet.sprite.Sprite(image)
    except:
        img = pyglet.text.Label(
            text="Can load file: " + path,
            font_size=16,
            x=100,
            y=100
        )
        window = pyglet.window.Window(600, 200, title)

    # on draw event
    @window.event
    def on_draw():
        window.clear()
        img.draw()

    @window.event
    def on_key_press(symbol, modifier):
        if symbol == pyglet.window.key.E:
            window.close()

    pyglet.app.run()