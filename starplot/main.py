import os
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time


def read_star_dat(dat_fp: Path) -> pd.DataFrame:
    df = pd.DataFrame()

    with dat_fp.open("rb") as f:
        # First 28 bytes contain following info:
        # Integer*4 STAR0=0    Subtract from star number to
        #                     get sequence number
        # Integer*4 STAR1=1    First star number in file
        # Integer*4 STARN=9110      Number of stars in file
        #                     -1=J2000
        # Integer*4 STNUM=1    0 if no star i.d. numbers are present
        #             1 if star i.d. numbers are in catalog file
        #             2 if star i.d. numbers are  in file
        # Logical*4 MPROP=1    1 if proper motion is included
        #             0 if no proper motion is included
        # Integer*4 NMAG=-1    Number of magnitudes present
        # Integer*4 NBENT=32    Number of bytes per star entry
        raw_header = f.read(28)
        star0, star1, starn, stnum, mprop, nmmag, nbent = struct.unpack(
            "<7i", raw_header
        )

        num_stars: int = int(abs(starn))
        for _ in range(num_stars):
            # Unpack each entry (32 bytes) in the following format:
            # Real*4 XNO		Catalog number of star
            # Real*8 SRA0		B1950 Right Ascension (radians)
            # Real*8 SDEC0		B1950 Declination (radians)
            # Character*2 IS		Spectral type (2 characters)
            # Integer*2 MAG		V Magnitude * 100
            # Real*4 XRPM		R.A. proper motion (radians per year)
            # Real*4 XDPM		Dec. proper motion (radians per year)
            raw_data = f.read(nbent)
            (
                xno,
                sra0,
                sdec0,
                spec_ch1,
                spec_ch2,
                mag,
                xrpm,
                xdpm,
            ) = struct.unpack("<f2d2ch2f", raw_data)

            # special processing
            xno = int(xno)
            spec = chr(int.from_bytes(spec_ch1)) + chr(int.from_bytes(spec_ch2))

            new_row = pd.DataFrame(
                data={
                    "NO": xno,
                    "RA": sra0,
                    "DE": sdec0,
                    "IS": spec,
                    "MAG": mag,
                    "XRPM": xrpm,
                    "XDPM": xdpm,
                },
                index=[xno],
            )
            df = pd.concat([df, new_row])

    return df


def compute_star_pos(star_df: pd.DataFrame):
    # Radius of celestial sphere
    R = 1000
    # Positions of stars in not-to-scale celestial sphere
    star_df["x"] = R * np.cos(star_df["DE"]) * np.cos(star_df["RA"])
    star_df["y"] = R * np.cos(star_df["DE"]) * np.sin(star_df["RA"])
    star_df["z"] = R * np.sin(star_df["DE"])

    # Get longitude and latitude from OS
    lon_deg = float(os.environ.get("longitude"))
    lat_deg = float(os.environ.get("latitude"))
    lat_rad = np.radians(lat_deg)

    # Get Greenwich apparent sidereal time from Astropy
    loc = EarthLocation(lat=lat_deg * u.deg, lon=lon_deg * u.deg)
    t = Time("2026-05-03 12:00:00", scale="utc", location=loc)
    gast = t.sidereal_time("apparent", "greenwich").hour

    # local hour angle
    ra_hour = star_df["RA"] * 12.0 / np.pi
    lha_deg = (gast - ra_hour) * 15 - lon_deg
    lha_rad = np.radians(lha_deg)

    star_df["altitude"] = np.arcsin(
        np.cos(lha_rad) * np.cos(star_df["DE"]) * np.cos(lat_rad)
        + np.sin(star_df["DE"]) * np.sin(lat_rad)
    )
    star_df["visibility_color"] = np.where(
        star_df["altitude"] > 0, "lime", "rgba(255, 255, 255, 0.2)"
    )


def plot_sphere(star_df: pd.DataFrame):
    # Plot an orthographic map of Earth, then a not-to-scale celestial sphere
    r_earth = 10
    u, v = np.mgrid[0 : 2 * np.pi : 50j, 0 : np.pi : 50j]
    earth_x = r_earth * np.cos(u) * np.sin(v)
    earth_y = r_earth * np.sin(u) * np.sin(v)
    earth_z = r_earth * np.cos(v)

    earth_sphere = go.Surface(
        x=earth_x,
        y=earth_y,
        z=earth_z,
        colorscale="Blues",
        showscale=False,
        opacity=0.9,
        name="Earth",
    )

    star_trace = go.Scatter3d(
        x=star_df["x"],
        y=star_df["y"],
        z=star_df["z"],
        mode="markers",
        marker=dict(size=2, color=star_df["visibility_color"], opacity=0.8),
        text=star_df["NO"],
        name="Stars",
    )

    fig = go.Figure(data=[earth_sphere, star_trace])

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )

    fig.show()


def main():
    dat_fp = Path("BSC5.bin")

    star_df = read_star_dat(dat_fp=dat_fp)

    compute_star_pos(star_df=star_df)

    plot_sphere(star_df=star_df)


if __name__ == "__main__":
    main()
