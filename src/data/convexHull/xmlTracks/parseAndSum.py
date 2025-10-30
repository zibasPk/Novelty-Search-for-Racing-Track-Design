import xml.etree.ElementTree as ET
import math
import json


def parse_and_check_track(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        x, y, theta = 0.0, 0.0, 0.0
        pos_tol = 1e-2   # meters
        angle_tol = 1e-3 # radians

        for section in root.findall(".//section"):
            name = section.get("name")
            if not name:
                continue

            if name.startswith("s"):
                # Straight
                length = 0.0
                for att in section:
                    if att.tag == "attnum" and att.get("name") in ("lg", "length"):
                        length = float(att.get("val", 0))
                dx = math.cos(theta) * length
                dy = math.sin(theta) * length
                x += dx
                y += dy

            elif name.startswith("c"):
                # Curve
                radius = arc_deg = direction = None
                for att in section:
                    if att.tag == "attnum" and att.get("name") == "radius":
                        radius = float(att.get("val", 0))
                    elif att.tag == "attnum" and att.get("name") == "arc":
                        arc_deg = float(att.get("val", 0))
                    elif att.tag == "attstr" and att.get("name") == "type":
                        direction = att.get("val")

                if radius is not None and arc_deg is not None and direction:
                    phi = math.radians(arc_deg)
                    phi = -abs(phi) if direction == "rgt" else abs(phi)

                    dx_loc = radius * math.sin(phi)
                    dy_loc = radius * (1 - math.cos(phi))
                    dx = math.cos(theta) * dx_loc - math.sin(theta) * dy_loc
                    dy = math.sin(theta) * dx_loc + math.cos(theta) * dy_loc

                    x += dx
                    y += dy
                    theta += phi

        # Normalize heading and compute errors
        two_pi = 2 * math.pi
        lap_count = round(theta / two_pi)
        angle_err = abs(theta - lap_count * two_pi)
        pos_err = math.hypot(x, y)
        closes = (pos_err <= pos_tol) and (angle_err <= angle_tol)

        print(f"Final position: x={x:.6f}, y={y:.6f}, theta={theta:.6f}")
        print(f"Position error: {pos_err:.6e}, Angle error: {angle_err:.6e}, Lap count: {lap_count}")
        print(f"Track closes: {closes}")
        
        # convert radians to degrees, position to meters and print again
        theta_deg = math.degrees(theta)
        pos_err_m = pos_err
        print(f"Final position (converted): x={x:.6f} m, y={y:.6f} m, theta={theta_deg:.6f} degrees")
        print(f"Position error (converted): {pos_err_m} m, Angle error: {math.degrees(angle_err)} degrees")

        return {
            "closes": closes,
            "finalPose": {"x": x, "y": y, "theta": theta},
            "posErr": pos_err,
            "angleErr": angle_err,
            "lapCount": lap_count
        }

    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None

if __name__ == "__main__":
    xml_file_path = "output_1725.xml"
    results = parse_and_check_track(xml_file_path)
    # append to json
    if results:
        with open("../../track_results.json", "a") as json_file:
            json.dump(results, json_file, indent=4)
