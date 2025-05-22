import bpy

def property_updated(self, context):
    """Called when target_property changes"""
    if self.target_object and self.target_property:
        try:
            obj = bpy.data.objects[self.target_object]
            prop_value = eval(f"obj.{self.target_property}")
            prop_type = type(prop_value).__name__
            if prop_type == "Vector":
                self.property_type = "VEC3"
            elif prop_type == "Matrix":
                self.property_type = "MAT4"
            elif prop_type == "float":
                self.property_type = "FLOAT"
            else:
                self.property_type = "Invalid"
        except:
            self.property_type = "Invalid"

class PointCloudUniforms(bpy.types.PropertyGroup):
    enabled: bpy.props.BoolProperty(name="Enabled", default=True)
    name: bpy.props.StringProperty(name="Name", default="")
    target_object: bpy.props.StringProperty(name="Target Object", default="")
    target_property: bpy.props.StringProperty(name="Target Property", default="", update=property_updated)
    ui_expanded: bpy.props.BoolProperty(name="Expanded", default=True)
    property_type: bpy.props.StringProperty(name="Property Type", default="")

    def evaluate_value(self):
        if self.target_object and self.target_property:
            try:
                obj = bpy.data.objects[self.target_object]
                prop_value = eval(f"obj.{self.target_property}")
                if self.property_type == "VEC3":
                    return (prop_value.x, prop_value.y, prop_value.z)
                elif self.property_type == "MAT4":
                    return (prop_value.row[0], prop_value.row[1], prop_value.row[2], prop_value.row[3])
                else:
                    return prop_value
            except:
                return None
