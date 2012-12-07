from PIL import Image
import colorsys
import os
import argparse

DEFAULT_THRESHOLD=1.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("background", help="New background to use.")
    parser.add_argument("target", help="Target image to be modified.")
    parser.add_argument("--threshold", help="Luminance threshold of target image [%s]" % DEFAULT_THRESHOLD, type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    
    bg = Image.open(args.background)
    tgt = Image.open(args.target)
    
    if bg.size != tgt.size:
        print "Background and target image must be the same size."
        exit(1)
        
    image = Image.new("RGB", bg.size)
        
    width,height = bg.size
    for x in range(width):
        for y in range(height):
            r,g,b = tgt.getpixel((x,y))
            h,s,v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
            if v < args.threshold:
                w_tgt = v/args.threshold
                w_bg = 1 - w_tgt                
                rb, gb, bb = bg.getpixel((x,y))
                hb, sb, vb =  colorsys.rgb_to_hsv(rb/255.0, gb/255.0, bb/255.0)
                v_new = v + w_bg*vb 
                r,g,b = colorsys.hsv_to_rgb(h,s,v_new)
                r,g,b = r*255, g*255, b*255
                r,g,b = (int(w_tgt*r + w_bg * rb), int(w_tgt*g + w_bg * gb), int(w_tgt*b + w_bg * bb))
            
            image.putpixel((x,y),(r,g,b))
    
    f, ext = os.path.splitext(args.target)
    image.save(f+".mod.png", "PNG")