from array import array
from math import log10, log
import struct
import zlib
import random

# how much each channel contributes to luminance
RGB_LUMINANCE = (0.2126, 0.7152, 0.0722)

DISPLAY_LUMINANCE_MAX = 200.0 # original value was 200.0

DERIVED_CONSTANT = 1.219

# formula from Ward "A Contrast-Based Scalefactor for Luminance Display"
SCALEFACTOR_NUMERATOR = 1.219 + (DISPLAY_LUMINANCE_MAX * 0.25) ** 0.4
SCALEFACTOR_ADJUST = 1.0 # value like 0.05 works well to retain highlights
GAMMA_ADJUST = 0.00 # value like -0.20 will adjust overall brightness of the image
GAMMA_ENCODE = 0.45 # orginal value was 0.45


class Image(object):

    def __init__(self, width, height, scalefactor_adjust=SCALEFACTOR_ADJUST, gamma_adjust=GAMMA_ADJUST):
        """
        initialize blank image.
        """
        self.width = width
        self.height = height
        self.scalefactor_adjust = scalefactor_adjust
        self.gamma_adjust = gamma_adjust
        self.data = array("d", [0]) * (width * height * 3)        
        self.lum_max = 0.0

    def _index(self, t):
        x, y, channel = t
        index = (x + ((self.height - 1 - y) * self.width)) * 3 + channel

        return min(max(index, 0), len(self.data) - 1)

    def __getitem__(self, t):
        return self.data[self._index(t)]

    def __setitem__(self, t, val):
        self.data[self._index(t)] = val

    def add_radiance(self, x, y, radiance):
        """
        add radiance (an RGB tuple) to given x, y position on image.
        """
        self[x, y, 0] += radiance[0]
        self[x, y, 1] += radiance[1]
        self[x, y, 2] += radiance[2]

    def calculate_scalefactor(self, iterations):
        """
        calculate the linear tone-mapping scalefactor for this image assuming
        the given number of iterations.
        """
        ## calculate the log-mean luminance of the image
        
        sum_of_logs = 0.0

        
        for x in range(self.width):
            for y in range(self.height):
                lum = self[x, y, 0] * RGB_LUMINANCE[0]
                lum += self[x, y, 1] * RGB_LUMINANCE[1]
                lum += self[x, y, 2] * RGB_LUMINANCE[2]
                lum /= iterations

                self.lum_max = max(self.lum_max, lum)
                
                sum_of_logs += log10(max(lum, 0.0001))

        log_mean_luminance = 10.0 ** (sum_of_logs / (self.height * self.width))
        
        # log_mean_luminance *= 1000

        ## calculate the scalefactor for linear tone-mapping

        # formula from Ward "A Contrast-Based Scalefactor for Luminance Display"

        scalefactor = (
            (SCALEFACTOR_NUMERATOR / (1.219 + log_mean_luminance ** 0.4)) ** 2.5
        ) / DISPLAY_LUMINANCE_MAX

        scalefactor *= self.scalefactor_adjust
        
        # print scalefactor
        return scalefactor

    def display_pixels(self, iterations):
        """
        iterate over each channel of each pixel in image returning
        gamma-corrected number scaled 0 - 1 (although not clipped to 1).
        """
        scalefactor = self.calculate_scalefactor(iterations)
                
        scaled_lum_max = self.lum_max * scalefactor/iterations 
        c = 1 / log(scaled_lum_max + 1)
        c2 = 1/(c ** (GAMMA_ENCODE+self.gamma_adjust))

        for value in self.data:
            scaled_value = max(value * scalefactor / iterations, 0) 
            updated_value = c*log10(scaled_value + 1)
            yield  c2 * updated_value ** (GAMMA_ENCODE + self.gamma_adjust)

    def save(self, filename, iterations):
        """
        save the image to given filename assuming the given number
        of iterations.
        """

        with open(filename, "wb") as f:
            f.write(struct.pack("8B", 137, 80, 78, 71, 13, 10, 26, 10))
            output_chunk(f, "IHDR", struct.pack("!2I5B", self.width, self.height, 8, 2, 0, 0, 0))
            compressor = zlib.compressobj()
            data = array("B")
            pixels = self.display_pixels(iterations)
            for y in range(self.height):
                data.append(0)
                for x in range(self.width):
                    for channel in range(3):
                        data.append(min(255, max(0, int(pixels.next() * 255.0 + 0.5))))
            compressed = compressor.compress(data.tostring())
            flushed = compressor.flush()
            output_chunk(f, "IDAT", compressed + flushed)
            output_chunk(f, "IEND", "")


def output_chunk(f, chunk_type, data):
    f.write(struct.pack("!I", len(data)))
    f.write(chunk_type)
    f.write(data)
    checksum = zlib.crc32(data, zlib.crc32(chunk_type))
    f.write(struct.pack("!i", checksum))
