# traj2grid.py
# Utility for mapping trajectory points to grid cells and grid vocabulary indices.

import numpy as np

_EPS_EDGE = 1e-12


class Traj2Grid:
    """
    Map (lon, lat[, time]) -> grid cell (gx, gy) -> vocabulary index.

    Supported point formats:
      - pandas Series / row (has .iloc)
      - dict {"lon": ..., "lat": ..., "time": ...}
      - tuple/list/ndarray: (lon, lat) or (lon, lat, time)
    """

    def __init__(
        self,
        m,
        n,
        min_lon,
        min_lat,
        max_lon,
        max_lat,
        grid2idx=None,
        out_of_bound="drop",  # "drop" or "clip"
    ):
        self.grid2idx = grid2idx if grid2idx else {}
        self.row_num = int(m)
        self.column_num = int(n)

        if self.row_num <= 0 or self.column_num <= 0:
            raise ValueError("row_num and column_num must be positive")

        self.min_lon = float(min_lon)
        self.min_lat = float(min_lat)
        self.max_lon = float(max_lon)
        self.max_lat = float(max_lat)

        self.h = (self.max_lat - self.min_lat) / self.row_num
        self.l = (self.max_lon - self.min_lon) / self.column_num

        if self.h <= 0 or self.l <= 0:
            raise ValueError("Grid size must be positive")

        if out_of_bound not in ("drop", "clip"):
            raise ValueError("out_of_bound must be 'drop' or 'clip'")

        self.out_of_bound = out_of_bound

    def _extract_lon_lat_time(self, point):
        """Return (lon, lat, time_or_None)."""

        # pandas Series / row
        if hasattr(point, "iloc"):
            if hasattr(point, "__contains__") and ("lon" in point) and ("lat" in point):
                lon = point["lon"]
                lat = point["lat"]
                t = point["time"] if ("time" in point) else None
                return float(lon), float(lat), (int(t) if t is not None else None)

            lon = point.iloc[0]
            lat = point.iloc[1]
            t = point.iloc[2] if len(point) > 2 else None
            return float(lon), float(lat), (int(t) if t is not None else None)

        # dict
        if isinstance(point, dict):
            lon = point["lon"]
            lat = point["lat"]
            t = point.get("time", None)
            return float(lon), float(lat), (int(t) if t is not None else None)

        # tuple / list / ndarray
        lon = point[0]
        lat = point[1]
        t = point[2] if len(point) > 2 else None
        return float(lon), float(lat), (int(t) if t is not None else None)

    def point2grid(self, point):
        """
        Convert a point to a grid cell (gx, gy).

        If out_of_bound="drop", return None when the point falls outside the grid.
        If out_of_bound="clip", clip the point to the nearest valid boundary cell.

        Boundary fix:
          lon == max_lon or lat == max_lat should fall into the last cell,
          not be treated as out-of-bound.
        """
        lon, lat, _ = self._extract_lon_lat_time(point)

        lon = min(lon, self.max_lon - _EPS_EDGE)
        lat = min(lat, self.max_lat - _EPS_EDGE)

        gx = int((lon - self.min_lon) // self.l)
        gy = int((lat - self.min_lat) // self.h)

        if gx < 0 or gx >= self.column_num or gy < 0 or gy >= self.row_num:
            if self.out_of_bound == "drop":
                return None
            gx = min(max(gx, 0), self.column_num - 1)
            gy = min(max(gy, 0), self.row_num - 1)

        return (gx, gy)

    def build_vocab(self, grid_count: dict, lower_bound=1, sort_mode="grid"):
        """
        Build grid vocabulary from a grid-count dictionary.

        Args:
            grid_count: {(gx, gy): count}
            lower_bound: minimum frequency threshold
            sort_mode:
                - "grid": sort by (gx, gy)
                - "count_desc": sort by count descending, then (gx, gy)
        """
        items = [(g, c) for g, c in grid_count.items() if c >= lower_bound]

        if sort_mode == "grid":
            items.sort(key=lambda x: (x[0][0], x[0][1]))
        elif sort_mode == "count_desc":
            items.sort(key=lambda x: (-x[1], x[0][0], x[0][1]))
        else:
            raise ValueError("sort_mode must be 'grid' or 'count_desc'")

        self.grid2idx.clear()
        for idx, (grid, _cnt) in enumerate(items):
            self.grid2idx[grid] = idx
        return self.grid2idx

    def set_vocab(self, grid2idx):
        """Set an externally prepared grid vocabulary."""
        self.grid2idx = grid2idx

    def convert1d(self, original_traj, diff=True, keep_time=True):
        """
        Convert a trajectory into:
          - traj_1d: [grid_idx, ...]
          - coord_traj: aligned coordinates

        Args:
            original_traj: iterable of points
            diff: if True, remove consecutive duplicate grid ids
            keep_time: if True and time exists, keep time in output coordinates
        """
        traj_1d = []
        coord_traj = []

        for p in original_traj:
            grid = self.point2grid(p)
            if grid is None:
                continue

            idx = self.grid2idx.get(grid, None)
            if idx is None:  # keep idx=0 as valid
                continue

            lon, lat, t = self._extract_lon_lat_time(p)
            if keep_time and t is not None:
                p_out = (lon, lat, int(t))
            else:
                p_out = (lon, lat)

            if diff:
                if (not traj_1d) or (idx != traj_1d[-1]):
                    traj_1d.append(idx)
                    coord_traj.append(p_out)
            else:
                traj_1d.append(idx)
                coord_traj.append(p_out)

        return traj_1d, coord_traj

    def draw_grid(self, grid_count: dict, file_name="grids.png"):
        """
        Draw a simple visualization of grid frequencies.
        """
        from PIL import Image

        img = Image.new("RGB", (self.column_num, self.row_num))

        vals = list(grid_count.values())
        mean = float(np.mean(vals)) if len(vals) else 0.0
        std = float(np.std(vals)) if len(vals) else 1.0
        std = std if std > 0 else 1.0

        for grid in self.grid2idx:
            percent = 50 * (grid_count.get(grid, 0) - mean) / std + 50
            if percent < 50:
                green = 255
                red = percent * 5.12
            else:
                red = 255
                green = 256 - (percent - 50) * 5.12

            color = (int(red), int(green), 0)
            img.putpixel((grid[0], grid[1]), color)

        img = img.resize((800, 800))
        img.save(file_name)