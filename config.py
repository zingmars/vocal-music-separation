import configparser

def prepare_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)

    # Set defaults
    config_get(config, 'logging', 'logfile', 'log.txt')
    config_get(config, 'logging', 'loglevel', 'INFO') #debug,info,warning,critical
    config_get(config, 'logging', 'logtype', 'console') #file/console

    config_get(config, 'song', 'sample_size', "22050") #Sample rate of the audio we will work with. If loaded audio doesn't match, it will be resampled.
    config_get(config, 'song', 'window_size', "1024") #We will get window size / 2 + 1 frequency bins to work with. 1024-1568 seems to be the perfect vales.
    config_get(config, 'song', 'hop_length', "256") #Size of each bin = hop size / sample size (in ms). The smaller it is, the more bins we get, but we don't need that much resolution.
    config_get(config, 'song', 'sample_length', "25") #Dictates how many frequency bins we give to the neural net for context. Less samples means more guesswork from the network, but also more samples from each song.

    config_get(config, 'model', 'save_history', "true") #Saves keras accuracy and loss history per epoch
    config_get(config, 'model', 'history_filename', "history.csv")

    with open(filename, 'w') as configfile: # If the file didn't exist, write default values to it
        config.write(configfile)
    return config

def config_get(config, section, key, default):
    try:
        config.get(section, key)
    except configparser.NoSectionError:
        config.add_section(section)
        config.set(section, key, default)
    except configparser.NoOptionError:
        config.set(section, key, default)
