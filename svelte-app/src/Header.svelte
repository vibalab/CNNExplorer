<script>
    import { Navbar, NavbarBrand, Nav, NavItem, NavLink} from 'sveltestrap';
</script>


<Navbar class="d-flex" color="secondary" light expand="xs" style="margin: -1px 0px 10px 0px;">
    <NavbarBrand href="/" style="color: black !important; font-weight: bold !important">CNN VIS에 HuggingFace를 곁들인...</NavbarBrand>
    <Nav className="ms-auto" navbar>
      <NavItem>
        <NavLink href="/" style="color: black !important;">VIBA</NavLink>
      </NavItem>
      <NavItem>
        <NavLink href="/" style="color: black !important;">GitHub</NavLink>
      </NavItem>
    </Nav>
  </Navbar>