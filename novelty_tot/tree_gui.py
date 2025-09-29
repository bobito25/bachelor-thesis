from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsLineItem, QApplication, QGraphicsTextItem, QMenu, QDialog, QVBoxLayout, QLabel, QPushButton
from PySide6.QtGui import QPen, QBrush, QWheelEvent, QPainter, QColor
from PySide6.QtCore import Qt

import argparse
import json
from tot import TreeOfThoughts

class TreeNode(QGraphicsRectItem):
    """A draggable node in the tree visualization."""
    def __init__(self, x, y, contents, depth=None, original_response=None, used_prompt=None, used_action_prompt=None, padding=10, max_width=200,
                 fixed_width=150, fixed_height=100):
            self.contents = contents
            self.is_expanded = False
            self.padding = padding
            self.max_width = max_width
            self.fixed_width = fixed_width
            self.fixed_height = fixed_height
            
            # Initialize with fixed dimensions
            super().__init__(-fixed_width / 2, -fixed_height / 2, fixed_width, fixed_height)  # Center node on (x, y)

            self.setPos(x, y)
            self.setBrush(QBrush(Qt.darkGray))
            self.setFlags(QGraphicsRectItem.ItemIsMovable | QGraphicsRectItem.ItemSendsScenePositionChanges)

            # Add text to the node (initially in collapsed state)
            self.text_item = QGraphicsTextItem(self)
            self.text_item.setDefaultTextColor(Qt.white)
            self.text_item.setHtml(contents)  # Using setHtml for better text rendering
            
            # Add clip item to contain text within boundaries
            self.clip_rect = QGraphicsRectItem(self)
            self.clip_rect.setRect(self.rect())
            self.clip_rect.setPen(Qt.NoPen)  # Make the clip rectangle invisible
            
            # Position text in the collapsed node
            self._adjust_text_position()

            self.depth = depth
            self.original_response = original_response
            self.used_prompt = used_prompt
            self.used_action_prompt = used_action_prompt if used_action_prompt else ""

    def _adjust_text_position(self):
        """Position the text in the center of the node based on current state."""
        if self.is_expanded:
            # In expanded mode, show full text
            self.text_item.setPos(self.rect().left() + self.padding, self.rect().top() + self.padding)
            
            # Remove clipping in expanded mode
            self.text_item.setParentItem(self)
        else:
            # In collapsed mode, set up clipping for the text
            self.text_item.setTextWidth(self.fixed_width - 2 * self.padding)
            
            # Create clip path to contain text within the node boundaries
            self.clip_rect.setRect(self.rect().adjusted(self.padding, self.padding, -self.padding, -self.padding))
            
            # Position text for best visibility in collapsed state
            text_height = min(self.fixed_height - 2 * self.padding, self.text_item.boundingRect().height())
            self.text_item.setPos(-self.fixed_width/2 + self.padding, -text_height/2)
            
            # Apply clipping by making the text item a child of the clip rectangle
            self.text_item.setParentItem(self.clip_rect)
            
            # Ensure the text doesn't render outside the node boundaries
            self.clip_rect.setFlag(QGraphicsRectItem.ItemClipsChildrenToShape, True)

    def expand(self):
        """Expand the node to show all content."""
        if self.is_expanded:
            return
            
        self.is_expanded = True
        
        # Calculate expanded size based on actual text content
        self.text_item.setTextWidth(self.max_width)
        self.text_item.adjustSize()
        text_width = self.text_item.boundingRect().width()
        text_height = self.text_item.boundingRect().height()
        
        # Set new dimensions with padding
        expanded_width = max(text_width + 2 * self.padding, self.fixed_width)
        expanded_height = max(text_height + 2 * self.padding, self.fixed_height)
        
        # Update rectangle size while keeping it centered
        self.setRect(-expanded_width / 2, -expanded_height / 2, expanded_width, expanded_height)
        self._adjust_text_position()
        
        # Update connected edges
        if hasattr(self.scene(), 'edges'):
            for edge in self.scene().edges:
                edge.update_position()

    def collapse(self):
        """Collapse the node to fixed size."""
        if not self.is_expanded:
            return
            
        self.is_expanded = False
        
        # Reset to fixed dimensions
        self.setRect(-self.fixed_width / 2, -self.fixed_height / 2, self.fixed_width, self.fixed_height)
        self._adjust_text_position()
        
        # Update connected edges
        if hasattr(self.scene(), 'edges'):
            for edge in self.scene().edges:
                edge.update_position()

    def itemChange(self, change, value):
        """Update edges when the node moves."""
        if change == QGraphicsRectItem.ItemPositionChange:
            # Update all connected edges
            if hasattr(self.scene(), 'edges'):
                for edge in self.scene().edges:
                    edge.update_position()
        return super().itemChange(change, value)

    def contextMenuEvent(self, event):
        """Handle right-click to show a context menu."""
        menu = QMenu()
        show_info_action = menu.addAction("Show Info")
        
        # Add expand/collapse option based on current state
        if self.is_expanded:
            expand_action = menu.addAction("Collapse Node")
        else:
            expand_action = menu.addAction("Expand Node")
        
        action = menu.exec(event.screenPos())
        if action == show_info_action:
            self.show_info_popup()
        elif action == expand_action:
            if self.is_expanded:
                self.collapse()
            else:
                self.expand()

    def show_info_popup(self):
        dialog = QDialog()
        dialog.setWindowTitle("Node Information")

        layout = QVBoxLayout(dialog)

        label_depth = QLabel(f"<b>Depth:</b> {self.depth}")
        label_depth.setTextFormat(Qt.RichText)
        label_depth.setWordWrap(True)
        layout.addWidget(label_depth)

        label_contents = QLabel(f"<b>Contents:</b> {self.text_item.toPlainText()}")
        label_contents.setTextFormat(Qt.RichText)
        label_contents.setWordWrap(True)
        layout.addWidget(label_contents)

        label_prompt = QLabel(f"<b>Used Prompt:</b> {self.used_prompt}")
        label_prompt.setTextFormat(Qt.RichText)
        label_prompt.setWordWrap(True)
        layout.addWidget(label_prompt)

        label_action_prompt = QLabel(f"<b>Used Action Prompt:</b> {self.used_action_prompt}")
        label_action_prompt.setTextFormat(Qt.RichText)
        label_action_prompt.setWordWrap(True)
        layout.addWidget(label_action_prompt)

        label_response = QLabel(f"<b>Original Response:</b> {self.original_response}")
        label_response.setTextFormat(Qt.RichText)
        label_response.setWordWrap(True)
        layout.addWidget(label_response)

        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button)

        # Dynamically adjust label widths when the dialog is resized
        def resize_labels():
            max_width = dialog.size().width() - 40  # Account for padding
            label_depth.setMaximumWidth(max_width)
            label_contents.setMaximumWidth(max_width)
            label_prompt.setMaximumWidth(max_width)
            label_action_prompt.setMaximumWidth(max_width)
            label_response.setMaximumWidth(max_width)

        dialog.resizeEvent = lambda event: (resize_labels(), super(QDialog, dialog).resizeEvent(event))
        resize_labels()  # Initial adjustment

        dialog.exec_()

class TreeEdge(QGraphicsLineItem):
    """A line connecting two tree nodes."""
    def __init__(self, node1, node2):
        super().__init__()
        self.node1 = node1
        self.node2 = node2
        self.setPen(QPen(Qt.black, 2))
        self.setZValue(-1)  # Ensure edges appear behind nodes and text
        self.update_position()

    def update_position(self):
        """Modify the line to connect the two nodes."""
        p1 = self.node1.scenePos()
        p2 = self.node2.scenePos()
        self.setLine(p1.x(), p1.y(), p2.x(), p2.y())

class TreeView(QGraphicsView):
    """Custom QGraphicsView to handle zooming and panning."""
    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, event: QWheelEvent):
        """Zoom in/out with mouse wheel."""
        zoom_factor = 1.2 if event.angleDelta().y() > 0 else 0.8
        self.scale(zoom_factor, zoom_factor)

def compute_subtree_width(state, x_spacing, padding=10, max_width=200):
    """
    Recursively calculate the total width of the subtree rooted at the given state,
    based on the actual sizes of the nodes.

    Args:
        state: The current state (TreeOfThoughts.State).
        x_spacing: Horizontal spacing between sibling nodes.
        padding: Padding around the text in each node.
        max_width: Maximum width for wrapping text in nodes.

    Returns:
        The total width of the subtree.
    """
    # Calculate the width of the current node
    text_item = QGraphicsTextItem(state.contents)
    text_item.setTextWidth(max_width)  # Wrap text to fit within max_width
    text_item.adjustSize()
    node_width = text_item.boundingRect().width() + 2 * padding

    if not state.children:
        return node_width

    # Calculate the total width of all child subtrees
    child_widths = [compute_subtree_width(child, x_spacing, padding, max_width) for child in state.children]
    total_width = sum(child_widths) + (len(child_widths) - 1) * x_spacing

    # Ensure the current node's width is considered if it's wider than the total child width
    return max(node_width, total_width)

def build_tot_tree_scene(root_state, x_spacing=120, y_spacing=120):
    """
    Build a QGraphicsScene from a Tree of Thoughts root state.

    Args:
        root_state: The root state from tot.TreeOfThoughts.
        x_spacing: Horizontal spacing between sibling nodes.
        y_spacing: Vertical spacing between parent and child.
        
    Returns:
        QGraphicsScene configured with nodes and edges.
    """
    scene = QGraphicsScene()
    scene.edges = []  # For updating edges when nodes move

    def add_state(state, x, y, x_spacing, y_spacing):
        node = TreeNode(x, y, state.contents, depth=state.depth, original_response=state.original_response, used_prompt=state.used_prompt, used_action_prompt=state.used_action_prompt)
        node.setToolTip(state.contents)
        scene.addItem(node)

        if state.children:
            total_width = compute_subtree_width(state, x_spacing)  # Use updated width computation
            start_x = x - (total_width - x_spacing) / 2
            offset = 0
            for child in state.children:
                child_width = compute_subtree_width(child, x_spacing)
                child_x = start_x + offset + child_width / 2
                child_y = y + y_spacing
                child_node = add_state(child, child_x, child_y, x_spacing, y_spacing)
                edge = TreeEdge(node, child_node)
                scene.addItem(edge)
                scene.edges.append(edge)
                offset += child_width + x_spacing
        return node

    add_state(root_state, 0, 0, x_spacing, y_spacing)
    return scene

def get_tree_view(root_state, x_spacing=120, y_spacing=120):
    """
    Create a QGraphicsView widget configured to display the tree derived from root_state.

    Args:
        root_state: The root state from tot.TreeOfThoughts.
        x_spacing: Horizontal spacing between sibling nodes.
        y_spacing: Vertical spacing between parent and child.
        
    Returns:
        A TreeView widget that can be shown.
    """
    scene = build_tot_tree_scene(root_state, x_spacing, y_spacing)
    return TreeView(scene)

def load_tree_from_file(file_path: str) -> TreeOfThoughts.State:
    """
    Load a tree of states from a JSON file and convert it into a TreeOfThoughts.State.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        TreeOfThoughts.State: The root state of the tree.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return TreeOfThoughts.State.from_dict(data)

def main():
    """
    Main function for the CLI script.
    """
    parser = argparse.ArgumentParser(description="Tree of Thoughts GUI")
    parser.add_argument(
        "file",
        type=str,
        help="Path to the JSON file containing the serialized tree of states."
    )
    args = parser.parse_args()

    try:
        root_state = load_tree_from_file(args.file)
        app = QApplication([])
        tree_view = get_tree_view(root_state)
        tree_view.show()
        app.exec()
    except Exception as e:
        print(f"Error loading or visualizing tree: {e}")

if __name__ == "__main__":
    main()
