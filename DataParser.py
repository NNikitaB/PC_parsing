from Functuons_strategies_for_scoring import get_array_strategies, is_numeric_frame, del_no_numeric, \
    drop_duplicates_all, drop_anomalies


class DataParser:
    """Parse data"""
    arrFunctions = get_array_strategies()

    def sub_parse(self, frame_pandas, list_indexes=None):
        if list_indexes is None or list_indexes == []:
            return drop_anomalies(drop_duplicates_all(frame_pandas))
        ln = len(self.arrFunctions)
        if not is_numeric_frame(frame_pandas):
            frame_pandas = del_no_numeric(frame_pandas)
            # raise Exception("Data Frame DO NOT NUMBER")
        for i in list_indexes:
            if i < ln:
                frame_pandas = self.arrFunctions[i](frame_pandas)
        return frame_pandas

