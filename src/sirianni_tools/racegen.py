"""Generate TORCS race setup for simulation and track export."""

__author__ = "Jacopo Sirianni"
__copyright__ = "Copyright 2015-2016, Jacopo Sirianni"
__license__ = "GPL"
__email__ = "jacopo.sirianni@mail.polimi.it"

import os

# Fixed bot configuration - these are always used for racing
RACING_BOTS = [
    ("lliaw", "7"),
    ("olethros", "2"),
    ("tita", "4"),
    ("inferno", "6")
]

BOT_ORDER_VARIATIONS = [[0,1,2,3],
                        [1,2,3,0],
                        [2,3,0,1],
                        [3,0,1,2],
                        [0,2,1,3],
                        [1,3,2,0],
                        [2,0,3,1],
                        [3,1,0,2],
                        [0,3,1,2],
                        [1,0,2,3],
                        [2,1,3,0],
                        [3,2,0,1],]

# Track exporter bot
TRACK_EXPORTER = ("trackexporter", "1")

def generate_track_export_xml(path):
    """Generate XML for track export (1 lap, trackexporter only)"""
    xml = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE params SYSTEM "../libs/tgf/params.dtd">

<params name="Quick Race" type="param" mode="mw">
    <section name="Tracks">
        <section name="1">
            <attstr name="name" val="output"/>
            <attstr name="category" val="road"/>
        </section>
    </section>

    <section name="Races">
        <section name="1">
            <attstr name="name" val="Quick Race"/>
        </section>
    </section>

    <section name="Quick Race">
        <attnum name="laps" val="1"/>
        <attstr name="type" val="race"/>
        <attstr name="starting order" val="random"/>
    </section>

    <section name="Drivers">
        <section name="1">
            <attnum name="idx" val="1"/>
            <attstr name="module" val="trackexporter"/>
        </section>
    </section>
</params>'''
    
    with open(path, "w") as f:
        f.write(xml)

def generate_race_xml(path, num_laps=10, iteration=0, change_order=True):
    """Generate XML for race simulation with all standard bots"""
    # Generate drivers section with all racing bots
    drivers_section = ""
    botOrder = BOT_ORDER_VARIATIONS[iteration % len(BOT_ORDER_VARIATIONS)] if change_order else BOT_ORDER_VARIATIONS[0]
    for i, bot_idx in enumerate(botOrder):
        bot_name, bot_id = RACING_BOTS[bot_idx]
        drivers_section += f'''
        <section name="{i+1}">
            <attnum name="idx" val="{bot_id}"/>
            <attstr name="module" val="{bot_name}"/>
        </section>'''

    xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE params SYSTEM "../libs/tgf/params.dtd">

<params name="Quick Race" type="param" mode="mw">
    <section name="Tracks">
        <section name="1">
            <attstr name="name" val="output"/>
            <attstr name="category" val="road"/>
        </section>
    </section>

    <section name="Races">
        <section name="1">
            <attstr name="name" val="Quick Race"/>
        </section>
    </section>

    <section name="Quick Race">
        <attnum name="laps" val="{num_laps}"/>
        <attstr name="type" val="race"/>
        <attstr name="starting order" val="random"/>
    </section>

    <section name="Drivers">{drivers_section}
    </section>
</params>'''
    
    with open(path, "w") as f:
        f.write(xml)
        
        
        
def generate_benchmark_xml(path):
    """Generate XML for benchmark race 1 bot (olethros) racing alone for 2 laps"""
    xml = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE params SYSTEM "../libs/tgf/params.dtd">

<params name="Quick Race" type="param" mode="mw">
    <section name="Tracks">
        <section name="1">
            <attstr name="name" val="output"/>
            <attstr name="category" val="road"/>
        </section>
    </section>

    <section name="Races">
        <section name="1">
            <attstr name="name" val="Quick Race"/>
        </section>
    </section>

    <section name="Quick Race">
        <attnum name="laps" val="2"/>
        <attstr name="type" val="race"/>
        <attstr name="starting order" val="random"/>
    </section>

    <section name="Drivers">
        <section name="1">
            <attnum name="idx" val="4"/>
            <attstr name="module" val="tita"/>
        </section>
    </section>
</params>'''
    
    with open(path, "w") as f:
        f.write(xml)