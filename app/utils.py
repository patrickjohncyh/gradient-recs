from typing import List
import plotly.express as px
from plotly.subplots import make_subplots
import os
from PIL import Image


def plotly_tsne(tsne_results_df, title, enable_legend=True, width=None):
    if isinstance(tsne_results_df, List):
        assert isinstance(title, List)
        fig = make_subplots(rows=1,cols=len(tsne_results_df), subplot_titles=title,
                            horizontal_spacing = 0.01)

    else:
        fig = make_subplots(rows=1,cols=1, subplot_titles=[title])
        tsne_results_df = [tsne_results_df]

    for idx, df in enumerate(tsne_results_df):
        fig_scatter = px.scatter(df,
                         x='x',
                         y='y',
                         color="Category",
                         color_discrete_sequence=px.colors.qualitative.Bold,
                         hover_data={
                             'x': False,
                             'y': False,
                             'Category': True
                         },)

        for _ in fig_scatter['data']:
            if idx == 0 and len(tsne_results_df)!=1:
                _.update(showlegend=False)
            fig.add_trace(_, row=1, col=idx+1)
        full_fig_scatter = fig_scatter.full_figure_for_development()
        xrange = full_fig_scatter.layout.xaxis.range
        yrange = full_fig_scatter.layout.yaxis.range
        fig.update_xaxes(range=xrange, row=1, col=idx + 1)
        fig.update_yaxes(range=yrange, row=1, col=idx + 1)


    fig.update_layout(
        margin={
            'b':0, 'l':0 ,'r':0, 't':30
        },
        legend={
            'title':'category'
        },
        paper_bgcolor='rgba(0,0,0,0)',
        autosize=False,
        showlegend=enable_legend,
        height=400,
        width=width or 700,
    )
    fig.update_yaxes(title=None, visible=True, showticklabels=False, showgrid=True, fixedrange=True)
    fig.update_xaxes(title=None, visible=True, showticklabels=False, showgrid=True, fixedrange=True)
    return fig


# https://stackoverflow.com/questions/41718892/pillow-resizing-a-gif
def resize_gif(path, save_as=None, scale=1):
    """
    Resizes the GIF to a given length:

    Args:
        path: the path to the GIF file
        save_as (optional): Path of the resized gif. If not set, the original gif will be overwritten.
        resize_to (optional): new size of the gif. Format: (int, int). If not set, the original GIF will be resized to
                              half of its size.
    """
    all_frames = extract_and_resize_frames(path, scale)

    if not save_as:
        save_as = path

    if len(all_frames) == 1:
        print("Warning: only 1 frame found")
        all_frames[0].save(save_as, optimize=True)
    else:
        all_frames[0].save(save_as, optimize=True, save_all=True, append_images=all_frames[1:], loop=1000)


def analyseImage(path):
    """
    Pre-process pass over the image to determine the mode (full or additive).
    Necessary as assessing single frames isn't reliable. Need to know the mode
    before processing all frames.
    """
    im = Image.open(path)
    results = {
        'size': im.size,
        'mode': 'full',
    }
    try:
        while True:
            if im.tile:
                tile = im.tile[0]
                update_region = tile[1]
                update_region_dimensions = update_region[2:]
                if update_region_dimensions != im.size:
                    results['mode'] = 'partial'
                    break
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return results


def extract_and_resize_frames(path, scale):
    """
    Iterate the GIF, extracting each frame and resizing them

    Returns:
        An array of all frames
    """
    mode = analyseImage(path)['mode']

    im = Image.open(path)

    resize_to = (im.size[0] // scale, im.size[1] // scale)

    i = 0
    p = im.getpalette()
    last_frame = im.convert('RGBA')

    all_frames = []

    try:
        while True:
            # print("saving %s (%s) frame %d, %s %s" % (path, mode, i, im.size, im.tile))

            '''
            If the GIF uses local colour tables, each frame will have its own palette.
            If not, we need to apply the global palette to the new frame.
            '''

            new_frame = Image.new('RGBA', im.size)

            '''
            Is this file a "partial"-mode GIF where frames update a region of a different size to the entire image?
            If so, we need to construct the new frame by pasting it on top of the preceding frames.
            '''
            if mode == 'partial':
                new_frame.paste(last_frame)

            new_frame.paste(im, (0, 0), im.convert('RGBA'))

            new_frame.thumbnail(resize_to, Image.ANTIALIAS)
            all_frames.append(new_frame)

            i += 1
            last_frame = new_frame
            im.seek(im.tell() + 1)
    except EOFError:
        pass

    return all_frames
