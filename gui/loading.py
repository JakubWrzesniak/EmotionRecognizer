import time

import pyglet

start_time = time.time()
name = "Creating new model"
i = 0
def loading(th):


    animSprite = None
    title = "Creating new model"
    animation = pyglet.image.load_animation('gui/img/loading.gif')
    animSprite = pyglet.sprite.Sprite(animation)

    w = animSprite.width
    h = animSprite.height

    window = pyglet.window.Window(w, h, title)

    r, g, b, alpha = 0.5, 0.5, 0.8, 0.5

    pyglet.gl.glClearColor(r, g, b, alpha)

    def title():
        global i
        global start_time
        global name
        if time.time() - start_time > 1:
            i = (i + 1) % 4
            if i == 0:
                name = "Creating new model"
            name += "."
            start_time = time.time()

        title = pyglet.text.Label(text=name,
                                    font_name='Solaris',
                                    font_size=20,
                                    x=w / 2,
                                    y=h - 20,
                                    anchor_x='center',
                                    anchor_y='center')
        return title

    @window.event
    def on_draw():
        window.clear()
        animSprite.draw()
        title().draw()

    @window.event
    def on_close():
        pass


    def exittt(dt):
        if th.is_alive():
            pass
        else:
            th.join()
            window.close()
            pyglet.app.exit()

    th.start()
    pyglet.clock.schedule_interval(func=exittt, interval=3.)
    pyglet.app.run()



