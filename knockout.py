from PIL import Image
import colorsys
import os
import argparse

DEFAULT_THRESHOLD=1.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="Target image to be modified.")
    parser.add_argument("--threshold", help="Luminance threshold of target image [%s]" % DEFAULT_THRESHOLD, type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    
    tgt = Image.open(args.target)
    mask = Image.new('L', tgt.size, color=0)
            
    image = Image.new("RGB", tgt.size)
        
    width,height = tgt.size
    for x in range(width):
        for y in range(height):
            r,g,b = tgt.getpixel((x,y))
            h,s,v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
            w_tgt = v
            w_bg = 1 - w_tgt       
            v_new = v + w_bg*0.5         
            r,g,b = colorsys.hsv_to_rgb(h,s,v_new)
            r,g,b = int(r*255), int(g*255), int(b*255)
            
            image.putpixel((x,y),(r,g,b))            
            mask.putpixel((x,y),w_tgt*255)
    
    image.putalpha(mask)
    f, ext = os.path.splitext(args.target)
    image.save(f+".ko.png", "PNG")