// Weight-calibration jig for the 4x4 tactile sensor: a template plate on legs that holds a sliding
// plunger over one sensing point, so the whole weight goes through a small flat tip onto that point.
//
//   weight
//   ======== tray (centring grooves for each weight's base)
//      ||    shaft slides in a hole of the plate
//   ===||=== template plate with guide sleeves, on 4 separate legs, NOT touching the sensor
//      ##    tip (3 x 3 mm, add 0.5-1 mm soft silicone under it)
//   -------- sensor
//
// Open in OpenSCAD (free), set the measured values below, pick `part`, F6 render, File > Export STL.
// Print order: "plunger" and "coupon" first, try the plunger shaft in the coupon holes, set hole_d to the
// hole it slides in freely without wobble, then print "plate".
// Weigh the printed plunger and pass it to weight_cal.py as --platen-g.

// ---------------- MEASURE ON YOUR SENSOR ----------------
pitch_x   = 10.0;   // centre-to-centre distance between sensing points along a row (mm)
pitch_y   = 10.0;   // ... between rows (mm)
n_rows    = 4;
n_cols    = 4;
sensor_t  = 1.5;    // total thickness of the sensor stack lying on the bench (mm)
sensor_w  = 60;     // sensor outline, width (mm): the legs stand outside it
sensor_h  = 60;     // sensor outline, height (mm)

// ---------------- PRINTER FIT ----------------
shaft_d   = 5.8;    // plunger shaft
hole_d    = 6.3;    // guide hole; FDM holes come out 0.2-0.4 mm small, check with the coupon
coupon_holes = [6.0, 6.2, 6.4, 6.6];

// ---------------- TEMPLATE PLATE ----------------
plate_t   = 3.0;    // plate thickness
guide_l   = 9.0;    // guide length = plate + a sleeve on top; long guide = small tilt, small side force
boss_d    = hole_d + 3.0;  // sleeve outer diameter (neighbouring sleeves may merge)
clearance = 3.0;    // gap between the sensor top and the plate underside
leg_d     = 8.0;
peg_d     = 3.0;    // the legs are separate parts with a peg into the plate corners
leg_gap   = 4.0;    // legs this far outside the sensor outline

// ---------------- PLUNGER ----------------
tip_w     = 3.0;    // flat tip, a bit larger than the 2 x 2 mm sensing point (+-0.5 mm alignment)
tip_h     = 1.5;
travel    = 3.0;    // free height between sleeve top and tray underside with the tip on the sensor
tray_d    = 50.0;   // larger than the largest weight's base
tray_t    = 2.5;
guide_rings = [22, 28, 38];  // base diameters of your weights (measure!), engraved as centring grooves
groove_w  = 0.8;
groove_depth = 0.6;

part = "all";       // "plate", "legs", "plunger", "coupon", "all"
$fn = 64;

// ---------------- derived ----------------
grid_w  = (n_cols - 1) * pitch_x;
grid_h  = (n_rows - 1) * pitch_y;
plate_w = max(sensor_w, grid_w + 2 * hole_d) + 2 * (leg_gap + leg_d);
plate_h = max(sensor_h, grid_h + 2 * hole_d) + 2 * (leg_gap + leg_d);
leg_h   = sensor_t + clearance;              // bench to plate underside
shaft_l = clearance + guide_l + travel - tip_h;

function leg_xy() = [for (sx = [-1, 1], sy = [-1, 1])
    [sx * (plate_w / 2 - leg_d / 2 - 1), sy * (plate_h / 2 - leg_d / 2 - 1)]];
function hole_xy(r, c) = [(c - (n_cols - 1) / 2) * pitch_x, ((n_rows - 1) / 2 - r) * pitch_y];  // R1 at +y

// Template in use orientation: legs on the bench (z = 0), plate on top. The notch marks the R1C1 corner.
module plate() {
    difference() {
        union() {
            translate([-plate_w / 2, -plate_h / 2, leg_h]) cube([plate_w, plate_h, plate_t]);
            for (r = [0 : n_rows - 1], c = [0 : n_cols - 1])   // guide sleeves
                translate([hole_xy(r, c)[0], hole_xy(r, c)[1], leg_h]) cylinder(d = boss_d, h = guide_l);
        }
        for (r = [0 : n_rows - 1], c = [0 : n_cols - 1])
            translate([hole_xy(r, c)[0], hole_xy(r, c)[1], leg_h - 1]) cylinder(d = hole_d, h = guide_l + 2);
        for (p = leg_xy())   // sockets for the separate legs
            translate([p[0], p[1], leg_h - 1]) cylinder(d = peg_d + 0.4, h = plate_t + 2);
        // R1C1 corner notch (top left when the legs are down and you read the sensor normally)
        translate([-plate_w / 2, plate_h / 2, leg_h - 1]) rotate([0, 0, -45]) cube([8, 8, plate_t + 2], center = true);
    }
}

// One leg: stands on the bench, its peg goes up into a plate corner.
module leg() {
    cylinder(d = leg_d, h = leg_h);
    translate([0, 0, leg_h - 0.01]) cylinder(d = peg_d, h = plate_t * 0.8);
}

// Plunger in use orientation: tip on the sensor at z = 0, tray on top.
module plunger() {
    difference() {
        union() {
            translate([-tip_w / 2, -tip_w / 2, 0]) cube([tip_w, tip_w, tip_h + 0.01]);
            translate([0, 0, tip_h]) cylinder(d = shaft_d, h = shaft_l + 0.01);
            translate([0, 0, tip_h + shaft_l]) cylinder(d = tray_d, h = tray_t);
        }
        for (d = guide_rings)
            translate([0, 0, tip_h + shaft_l + tray_t - groove_depth])
                difference() {
                    cylinder(d = d + groove_w, h = groove_depth + 1);
                    translate([0, 0, -1]) cylinder(d = d - groove_w, h = groove_depth + 3);
                }
    }
}

// Fit test: candidate hole sizes (as long as the real guide, so you feel the real fit); one notch per step (1 = smallest).
module coupon() {
    n = len(coupon_holes);
    difference() {
        translate([-n * 6, -7, 0]) cube([n * 12, 14, guide_l]);
        for (i = [0 : n - 1]) {
            translate([(i - (n - 1) / 2) * 12, 0, -1]) cylinder(d = coupon_holes[i], h = guide_l + 2);
            for (k = [0 : i])
                translate([(i - (n - 1) / 2) * 12 + (k - i / 2) * 2, -7, -1]) cube([1, 2, guide_l * 2 + 2], center = true);
        }
    }
}

// Print orientations: plate as used (flat underside on the bed), legs standing, plunger upside down (tray on the bed).
if (part == "plate")   translate([0, 0, -leg_h]) plate();    // flat underside on the bed, sleeves up
if (part == "legs")    for (i = [0 : 3]) translate([i * (leg_d + 4), 0, 0]) leg();
if (part == "plunger") translate([0, 0, tip_h + shaft_l + tray_t]) rotate([180, 0, 0]) plunger();
if (part == "coupon")  coupon();
if (part == "all") {   // assembled view for checking, not for printing
    color("lightgray") plate();
    color("gray") for (p = leg_xy()) translate([p[0], p[1], 0]) leg();
    color("steelblue") translate([hole_xy(1, 1)[0], hole_xy(1, 1)[1], sensor_t]) plunger();
    color("orange", 0.5) translate([-sensor_w / 2, -sensor_h / 2, 0]) cube([sensor_w, sensor_h, sensor_t]);
}
