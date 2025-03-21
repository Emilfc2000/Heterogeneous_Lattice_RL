import numpy
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import ezdxf
from shapely.geometry import Polygon, MultiPolygon

def sample_new_point(origin_square, length_halfsquare, subidx):
    dx, dy = subidx % 2, subidx // 2
    offset = length_halfsquare * numpy.array([dx, dy], dtype=float)
    random_offset = numpy.array([numpy.random.random(), numpy.random.random()])
    return origin_square + random_offset * length_halfsquare + offset


def subdivide_square(origin_square, length_square, seeds, density_func):
    length_halfsquare = 0.5 * length_square
    rho = density_func(origin_square + length_halfsquare)
    target_seeds = (length_square ** 2) * rho
    if target_seeds <= 4:
        # 1st case: the cell is a leaf
        shuffled_idx = numpy.random.permutation(4)
        min_samples = int(numpy.floor(target_seeds))
        proba_last = target_seeds - min_samples
        for i in range(min_samples):
            seeds.append(sample_new_point(origin_square, length_halfsquare, shuffled_idx[i]))
        if numpy.random.random() <= proba_last and min_samples < 4:
            seeds.append(sample_new_point(origin_square, length_halfsquare, shuffled_idx[min_samples]))
    else:
        # 2nd case: recursive call
        for delta in numpy.ndindex(2, 2):
            offset = numpy.array(delta, dtype=float)
            origin_subsquare = origin_square + offset * length_halfsquare
            subdivide_square(origin_subsquare, length_halfsquare, seeds, density_func)


def plot_seeds(seeds, extent):
    seeds_x = [s[0] for s in seeds]
    seeds_y = [s[1] for s in seeds]
    plt.scatter(seeds_x, seeds_y, s=0.5)
    plt.xlim([0, extent[0]])
    plt.ylim([0, extent[1]])
    plt.axis('equal')
    plt.show()
    return numpy.column_stack([seeds_x,seeds_y])


def generate_seeds(coarse_level_length, extent):
    def density_func(point):
        # grading in x direction
        seed_density_factor = 2
        grad_density = (point[0] / extent[0]) * seed_density_factor # seeds / mm^2
        const_density = seed_density_factor/5
        return grad_density #Change to grad_density if output should have density gradiant

    numpy.random.seed(1)
    seeds = []
    for origin_x in numpy.arange(0.0, extent[0], coarse_level_length):
        for origin_y in numpy.arange(0.0, extent[1], coarse_level_length):
            origin_square_coarse = numpy.array([origin_x, origin_y], dtype=float)
            subdivide_square(origin_square_coarse, coarse_level_length, seeds, density_func)

    return seeds


def plot_voronoi(seeds,thickness):
    vor = Voronoi(vor_seeds)
    voronoi_plot_2d(vor, line_width=thickness*6)
    plt.axis('scaled')
    plt.xlim([0, extent[0]])
    plt.ylim([0, extent[1]])
    plt.show()
    return vor


def create_voronoi_polygons(vor, extent):
    """
    Create a list of polygons corresponding to the Voronoi cells within the given bounds.
    """
    polygons = []
    for region_idx in vor.regions:
        if not -1 in region_idx and region_idx != []:
            # Retrieve vertices of the Voronoi cell
            polygon_vertices = [vor.vertices[i] for i in region_idx]
            polygon = Polygon(polygon_vertices)
            
            # Clip the polygon to stay within the bounding box
            bounding_box = Polygon([(0, 0), (extent[0], 0), (extent[0], extent[1]), (0, extent[1])])
            clipped_polygon = polygon.intersection(bounding_box)
            
            if not clipped_polygon.is_empty:
                polygons.append(clipped_polygon)
    
    return polygons


def create_offset_voronoi_edges(vor, thickness, extent):
    """
    Offset the Voronoi edges by a specified thickness to generate a lattice.
    The resulting polygons should form continuous edges.
    """
    from shapely.ops import unary_union
    from shapely.geometry import LineString, Polygon, box

    polygons = []
    bounding_box = box(0, 0, extent[0], extent[1])

    for ridge in vor.ridge_vertices:
        if -1 not in ridge:  # Skip ridges with infinite points
            p1, p2 = vor.vertices[ridge]

            line = LineString([p1, p2])
            offset_line = line.buffer(thickness / 2, cap_style=2)  # Offset on both sides

            # Clip with bounding box to avoid going out of bounds
            clipped_polygon = offset_line.intersection(bounding_box)

            # Add the resulting polygon (only if valid)
            if isinstance(clipped_polygon, Polygon):
                polygons.append(clipped_polygon)
            elif isinstance(clipped_polygon, MultiPolygon):
                for subpolygon in clipped_polygon.geoms:
                    polygons.append(subpolygon)

    # Merge all polygons into a unified geometry
    merged_polygon = unary_union(polygons)

    return merged_polygon


def save_data_2D_CAD(merged_polygon):
    """
    Save the unified Voronoi lattice as DXF format, including all internal edges.
    """
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    # Export the exterior and interior boundaries of the merged polygon
    if isinstance(merged_polygon, Polygon):
        coords = merged_polygon.exterior.coords
        for i in range(len(coords) - 1):
            msp.add_line(start=coords[i], end=coords[i + 1])

        # Handle any holes (interior polygons)
        for interior in merged_polygon.interiors:
            coords = interior.coords
            for i in range(len(coords) - 1):
                msp.add_line(start=coords[i], end=coords[i + 1])
    elif isinstance(merged_polygon, MultiPolygon):
        for polygon in merged_polygon.geoms:
            coords = polygon.exterior.coords
            for i in range(len(coords) - 1):
                msp.add_line(start=coords[i], end=coords[i + 1])

            # Handle any holes (interior polygons)
            for interior in polygon.interiors:
                coords = interior.coords
                for i in range(len(coords) - 1):
                    msp.add_line(start=coords[i], end=coords[i + 1])

    # Save the DXF document
    doc.saveas("voronoi_lattice_thickened.dxf")


    
def is_within_bounds(p1, p2, xmin, xmax, ymin, ymax):
    x1, y1 = p1
    x2, y2 = p2
    return (xmin <= x1 <= xmax and ymin <= y1 <= ymax) and \
           (xmin <= x2 <= xmax and ymin <= y2 <= ymax)
          
            
if __name__ == "__main__":
    coarse_level_length = 20  # (mm)
    extent = numpy.array([20.0, 20.0], dtype=float)  # (mm)
    thickness = 0.5  # (mm) thickness of the Voronoi lattice edges

    # Generate Voronoi seeds and compute Voronoi diagram
    seeds = generate_seeds(coarse_level_length, extent)
    vor_seeds = plot_seeds(seeds, extent)
    vor = Voronoi(vor_seeds)
    plot_voronoi(vor_seeds,thickness)

    # Create polygons that represent thickened Voronoi edges
    polygons = create_offset_voronoi_edges(vor, thickness, extent)

    # Save the thickened lattice as DXF
    save_data_2D_CAD(polygons)
